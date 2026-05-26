
// ═══════════════════════════════════════════════════════════════
// PIXELCOMPOSER ENGINE v3 — Structure-Aware Perceptual Fusion
// Pipeline:
//   Lanczos → linear RGB → analyse (edge+var+grad+saliency)
//   → OKLAB conversion → structure protection map
//   → adaptive weights (coherence-guided, spatially smoothed)
//   → Laplacian pyramid fusion (gradient-coherence weighted)
//   → collapse → ACES tone map → low-freq lock
//   → edge-gated sharpen → dither → sRGB
//
// Key upgrades over v2:
//   • OKLAB replaces HSL — perceptually stable color
//   • Gradient coherence field — stops texture smearing
//   • Structure protection map — preserves edges/silhouettes
//   • Saturation clamp (max 0.85) — prevents neon bloom
//   • All clamping deferred to final encode stage
// ═══════════════════════════════════════════════════════════════

// ── sRGB ↔ Linear ─────────────────────────────────────────────
function toLinear(v){v/=255;return v<=0.04045?v/12.92:Math.pow((v+0.055)/1.055,2.4);}
function toSRGB(v){v=Math.max(0,Math.min(1,v));return(v<=0.0031308?v*12.92*255:(1.055*Math.pow(v,1/2.4)-0.055)*255);}
function toLinearBuf(rgba,n){
  const lin=new Float32Array(n*3);
  for(let i=0;i<n;i++){lin[i*3]=toLinear(rgba[i*4]);lin[i*3+1]=toLinear(rgba[i*4+1]);lin[i*3+2]=toLinear(rgba[i*4+2]);}
  return lin;
}
function toSRGBBuf(lin,n){
  const out=new Uint8ClampedArray(n*4);
  for(let i=0;i<n;i++){out[i*4]=toSRGB(lin[i*3])|0;out[i*4+1]=toSRGB(lin[i*3+1])|0;out[i*4+2]=toSRGB(lin[i*3+2])|0;out[i*4+3]=255;}
  return out;
}

// ── OKLAB color space ─────────────────────────────────────────
// Far more perceptually uniform than HSL.
// Hue blending in OKLab is stable — no neon shifts, no saturation blooms.
function linearToOklab(r,g,b){
  const l=0.4122214708*r+0.5363325363*g+0.0514459929*b;
  const m=0.2119034982*r+0.6806995451*g+0.1073969566*b;
  const s=0.0883024619*r+0.2817188376*g+0.6299787005*b;
  const lc=Math.cbrt(l),mc=Math.cbrt(m),sc=Math.cbrt(s);
  return[
    0.2104542553*lc+0.7936177850*mc-0.0040720468*sc,
    1.9779984951*lc-2.4285922050*mc+0.4505937099*sc,
    0.0259040371*lc+0.7827717662*mc-0.8086757660*sc
  ];
}
function oklabToLinear(L,a,b){
  const lc=L+0.3963377774*a+0.2158037573*b;
  const mc=L-0.1055613458*a-0.0638541728*b;
  const sc=L-0.0894841775*a-1.2914855480*b;
  const lv=lc*lc*lc,mv=mc*mc*mc,sv=sc*sc*sc;
  return[
    +4.0767416621*lv-3.3077115913*mv+0.2309699292*sv,
    -1.2684380046*lv+2.6097574011*mv-0.3413193965*sv,
    -0.0041960863*lv-0.7034186147*mv+1.7076147010*sv
  ];
}

// ── Lanczos-3 ─────────────────────────────────────────────────
function lanczosK(x,a){if(x===0)return 1;if(Math.abs(x)>=a)return 0;const px=Math.PI*x;return(a*Math.sin(px)*Math.sin(px/a))/(px*px);}
function lanczosResize(src,sw,sh,dw,dh){
  const a=3,tmp=new Float32Array(dw*sh*4),out=new Uint8ClampedArray(dw*dh*4),xR=sw/dw;
  for(let y=0;y<sh;y++){for(let x=0;x<dw;x++){
    const sx=(x+0.5)*xR-0.5,x0=Math.floor(sx)-a+1;
    let r=0,g=0,b=0,al=0,ws=0;
    for(let k=x0;k<=x0+2*a-1;k++){const kv=lanczosK(sx-k,a),px=Math.max(0,Math.min(sw-1,k)),si=(y*sw+px)*4;r+=src[si]*kv;g+=src[si+1]*kv;b+=src[si+2]*kv;al+=src[si+3]*kv;ws+=kv;}
    const di=(y*dw+x)*4;tmp[di]=ws?r/ws:0;tmp[di+1]=ws?g/ws:0;tmp[di+2]=ws?b/ws:0;tmp[di+3]=ws?al/ws:255;
  }}
  const yR=sh/dh;
  for(let y=0;y<dh;y++){for(let x=0;x<dw;x++){
    const sy=(y+0.5)*yR-0.5,y0=Math.floor(sy)-a+1;
    let r=0,g=0,b=0,ws=0;
    for(let k=y0;k<=y0+2*a-1;k++){const kv=lanczosK(sy-k,a),py=Math.max(0,Math.min(sh-1,k)),si=(py*dw+x)*4;r+=tmp[si]*kv;g+=tmp[si+1]*kv;b+=tmp[si+2]*kv;ws+=kv;}
    const di=(y*dw+x)*4;out[di]=Math.max(0,Math.min(255,ws?r/ws:0));out[di+1]=Math.max(0,Math.min(255,ws?g/ws:0));out[di+2]=Math.max(0,Math.min(255,ws?b/ws:0));out[di+3]=255;
  }}
  return out;
}

// ── Gaussian blur — 3ch Float32 ───────────────────────────────
function gaussBlur(data,w,h,sigma){
  const r=Math.max(1,Math.ceil(3*sigma))|0,ks=2*r+1,k=new Float32Array(ks);
  let s=0;for(let i=0;i<ks;i++){const x=i-r;k[i]=Math.exp(-(x*x)/(2*sigma*sigma));s+=k[i];}
  for(let i=0;i<ks;i++)k[i]/=s;
  const tmp=new Float32Array(w*h*3),out=new Float32Array(w*h*3);
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){
    let r0=0,g0=0,b0=0;
    for(let ki=-r;ki<=r;ki++){const px=Math.max(0,Math.min(w-1,x+ki)),si=(y*w+px)*3,kv=k[ki+r];r0+=data[si]*kv;g0+=data[si+1]*kv;b0+=data[si+2]*kv;}
    const di=(y*w+x)*3;tmp[di]=r0;tmp[di+1]=g0;tmp[di+2]=b0;
  }}
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){
    let r0=0,g0=0,b0=0;
    for(let ki=-r;ki<=r;ki++){const py=Math.max(0,Math.min(h-1,y+ki)),si=(py*w+x)*3,kv=k[ki+r];r0+=tmp[si]*kv;g0+=tmp[si+1]*kv;b0+=tmp[si+2]*kv;}
    const di=(y*w+x)*3;out[di]=r0;out[di+1]=g0;out[di+2]=b0;
  }}
  return out;
}

// ── Gaussian blur — 1ch Float32 ──────────────────────────────
function gaussBlur1ch(data,w,h,sigma){
  const r=Math.max(1,Math.ceil(3*sigma))|0,ks=2*r+1,k=new Float32Array(ks);
  let s=0;for(let i=0;i<ks;i++){const x=i-r;k[i]=Math.exp(-(x*x)/(2*sigma*sigma));s+=k[i];}
  for(let i=0;i<ks;i++)k[i]/=s;
  const tmp=new Float32Array(w*h),out=new Float32Array(w*h);
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){let v=0;for(let ki=-r;ki<=r;ki++){v+=data[y*w+Math.max(0,Math.min(w-1,x+ki))]*k[ki+r];}tmp[y*w+x]=v;}}
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){let v=0;for(let ki=-r;ki<=r;ki++){v+=tmp[Math.max(0,Math.min(h-1,y+ki))*w+x]*k[ki+r];}out[y*w+x]=v;}}
  return out;
}

function lum(lin,i){return 0.2126*lin[i*3]+0.7152*lin[i*3+1]+0.0722*lin[i*3+2];}


// ── BILATERAL FILTER — pre-fusion denoising ───────────────────
// Removes JPEG artifacts and noise BEFORE pyramid decomposition.
// Unlike Gaussian blur, bilateral preserves edges while smoothing
// flat regions. This prevents artifacts from being amplified
// through the pyramid levels.
// sigma_s = spatial radius, sigma_r = range (color) sigma
function bilateralFilter(data, w, h, sigmaS, sigmaR) {
  const out = new Float32Array(data.length);
  const r = Math.max(1, Math.ceil(2 * sigmaS)) | 0;
  const ss2 = 2 * sigmaS * sigmaS;
  const sr2 = 2 * sigmaR * sigmaR;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 3;
      const rc = data[p], gc = data[p+1], bc = data[p+2];
      let sumR=0,sumG=0,sumB=0,sumW=0;
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const ny = Math.max(0, Math.min(h-1, y+dy));
          const nx = Math.max(0, Math.min(w-1, x+dx));
          const q = (ny * w + nx) * 3;
          const rq=data[q],gq=data[q+1],bq=data[q+2];
          // Spatial weight
          const spatialW = Math.exp(-(dx*dx+dy*dy)/ss2);
          // Range weight — color similarity
          const dr=rc-rq,dg=gc-gq,db=bc-bq;
          const rangeW = Math.exp(-(dr*dr+dg*dg+db*db)/sr2);
          const w2 = spatialW * rangeW;
          sumR+=rq*w2;sumG+=gq*w2;sumB+=bq*w2;sumW+=w2;
        }
      }
      out[p]  = sumR/sumW;
      out[p+1]= sumG/sumW;
      out[p+2]= sumB/sumW;
    }
  }
  return out;
}

// ── CROSS-SCALE COHERENCE ─────────────────────────────────────
// Real detail persists across pyramid levels.
// Noise appears only in the finest level and disappears quickly.
// This map identifies which pixels have "real" high-frequency
// content vs compression/interpolation artifacts.
// Returns Float32Array [0-1] per pixel — 1 = coherent real detail
function crossScaleCoherence(pyr, w, h) {
  const n = w * h;
  const coherence = new Float32Array(n);
  // Compare fine level magnitude against coarser levels
  // Real edges: strong at level 0, still present at level 1-2
  // Noise: strong at level 0, near-zero at level 1
  const l0 = pyr[0].lap; // finest
  const l1 = pyr[1].lap; // one level coarser
  const l2 = pyr[2].lap; // two levels coarser
  // Upsample l1 and l2 to full res for comparison
  const l1up = upsample(l1, pyr[1].w, pyr[1].h, w, h);
  const l2up = upsample(l2, pyr[2].w, pyr[2].h, w, h);
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    // Magnitude at each level (use luminance channel for speed)
    const m0 = Math.abs(0.2126*l0[p]+0.7152*l0[p+1]+0.0722*l0[p+2]);
    const m1 = Math.abs(0.2126*l1up[p]+0.7152*l1up[p+1]+0.0722*l1up[p+2]);
    const m2 = Math.abs(0.2126*l2up[p]+0.7152*l2up[p+1]+0.0722*l2up[p+2]);
    // Persistence ratio: if detail survives into coarser levels it's real
    // Weighted: level 1 persistence counts more than level 2
    const persist = (m1 * 0.6 + m2 * 0.4) / (m0 + 1e-8);
    // Clamp to [0,1] — high persistence = real detail
    coherence[i] = Math.min(1, persist * 3.0);
  }
  // Smooth slightly to avoid pixel-level noise in the coherence map
  return gaussBlur1ch(coherence, w, h, 0.6);
}

// ── SCENE ANALYSIS ────────────────────────────────────────────
// Returns continuous soft fields — no hard classes.
// NEW: gradX, gradY (oriented gradients), saliency, protect map
function analyseScene(lin,w,h){
  const n=w*h;
  const edgeMap=new Float32Array(n);
  const varMap=new Float32Array(n);
  const lumMap=new Float32Array(n);
  const gradX=new Float32Array(n);  // oriented gradient X
  const gradY=new Float32Array(n);  // oriented gradient Y
  const protect=new Float32Array(n);// structure protection [0-1]

  for(let i=0;i<n;i++)lumMap[i]=lum(lin,i);

  // Sobel — store both magnitude and direction
  for(let y=1;y<h-1;y++){for(let x=1;x<w-1;x++){
    const i=y*w+x;
    const gx=(-lumMap[(y-1)*w+(x-1)]+lumMap[(y-1)*w+(x+1)]-2*lumMap[y*w+(x-1)]+2*lumMap[y*w+(x+1)]-lumMap[(y+1)*w+(x-1)]+lumMap[(y+1)*w+(x+1)]);
    const gy=(-lumMap[(y-1)*w+(x-1)]-2*lumMap[(y-1)*w+x]-lumMap[(y-1)*w+(x+1)]+lumMap[(y+1)*w+(x-1)]+2*lumMap[(y+1)*w+x]+lumMap[(y+1)*w+(x+1)]);
    const mag=Math.sqrt(gx*gx+gy*gy);
    edgeMap[i]=mag;
    gradX[i]=gx;
    gradY[i]=gy;
  }}
  let eMax=0;for(let i=0;i<n;i++)if(edgeMap[i]>eMax)eMax=edgeMap[i];
  if(eMax>0)for(let i=0;i<n;i++){edgeMap[i]/=eMax;gradX[i]/=(eMax+1e-8);gradY[i]/=(eMax+1e-8);}

  // Local variance 5x5
  for(let y=2;y<h-2;y++){for(let x=2;x<w-2;x++){
    let sum=0,sum2=0,cnt=0;
    for(let dy=-2;dy<=2;dy++){for(let dx=-2;dx<=2;dx++){const v=lumMap[(y+dy)*w+(x+dx)];sum+=v;sum2+=v*v;cnt++;}}
    const mean=sum/cnt;varMap[y*w+x]=Math.max(0,sum2/cnt-mean*mean);
  }}
  let vMax=0;for(let i=0;i<n;i++)if(varMap[i]>vMax)vMax=varMap[i];
  if(vMax>0)for(let i=0;i<n;i++)varMap[i]/=vMax;

  // Saliency — center bias + edge density
  // Combines distance from image center with local edge energy
  const cx=w/2,cy=h/2,maxDist=Math.sqrt(cx*cx+cy*cy);
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){
    const i=y*w+x;
    const distFactor=1-Math.sqrt((x-cx)**2+(y-cy)**2)/maxDist;
    // Saliency = center bias * edge presence
    const sal=distFactor*0.4+edgeMap[i]*0.6;
    // Structure protection = edge + saliency — what must be preserved
    protect[i]=Math.min(1,edgeMap[i]*0.6+sal*0.4);
  }}

  // Smooth protection map — hard edges cause block artifacts
  const protectSmooth=gaussBlur1ch(protect,w,h,1.5);

  // Percentile thresholds for luminance
  const step=Math.max(1,Math.floor(n/2048));const smp=[];
  for(let i=0;i<n;i+=step)smp.push(lumMap[i]);smp.sort((a,b)=>a-b);
  const p15=smp[Math.floor(smp.length*0.15)]||0.1;
  const p85=smp[Math.floor(smp.length*0.85)]||0.9;

  // Continuous tone field
  const toneField=new Float32Array(n);
  for(let i=0;i<n;i++)toneField[i]=1-Math.abs(lumMap[i]-0.5)*2;

  return{edgeMap,varMap,lumMap,gradX,gradY,protect:protectSmooth,
    structure:edgeMap,texture:varMap,toneField,p15,p85};
}

// ── GRADIENT COHERENCE ────────────────────────────────────────
// Measures how well edge directions agree between A and B.
// High coherence = edges align = blend freely
// Low coherence  = texture conflict = protect both, blend less
function computeGradientCoherence(aA,aB,n){
  const coherence=new Float32Array(n);
  for(let i=0;i<n;i++){
    // Dot product of normalised gradient vectors
    const dot=aA.gradX[i]*aB.gradX[i]+aA.gradY[i]*aB.gradY[i];
    // Map from [-1,1] to [0,1] — 1 means perfect alignment
    coherence[i]=(dot+1)*0.5;
  }
  return coherence;
}

// ── ADAPTIVE WEIGHT MAPS ──────────────────────────────────────
// Incorporates: edge-aware, variance, luminance contrast,
//               gradient coherence, structure protection
function computeWeightMaps(aA,aB,w,h,mode,params){
  const n=w*h;
  const wA=new Float32Array(n);
  const wB=new Float32Array(n);
  const lowSrc=params.lowSrc||'A';

  // Gradient coherence field
  const coherence=computeGradientCoherence(aA,aB,n);

  for(let i=0;i<n;i++){
    const eA=aA.edgeMap[i],eB=aB.edgeMap[i];
    const vA=aA.varMap[i], vB=aB.varMap[i];
    const lA=aA.lumMap[i], lB=aB.lumMap[i];
    const pA=aA.protect[i],pB=aB.protect[i];
    const coh=coherence[i]; // 0=conflict, 1=aligned

    const eT=eA+eB+1e-8,vT=vA+vB+1e-8;
    const lcA=Math.abs(lA-0.5),lcB=Math.abs(lB-0.5),lcT=lcA+lcB+1e-8;
    const eWA=eA/eT,eWB=eB/eT,vWA=vA/vT,vWB=vB/vT,lcWA=lcA/lcT,lcWB=lcB/lcT;

    let wa,wb;

    if(mode==='frequency'){
      if(lowSrc==='A'){wa=0.5+0.3*(lcWA-0.5)-0.2*eWB;wb=1-wa;}
      else{wb=0.5+0.3*(lcWB-0.5)-0.2*eWA;wa=1-wb;}
    }else if(mode==='detail'){
      wb=0.35+0.4*eWB+0.25*vWB;wa=1-wb;
    }else if(mode==='soft_light'||mode==='overlay'){
      wa=0.35+0.3*lcWA+0.35*vWA;wb=0.35+0.3*lcWB+0.35*vWB;
    }else if(mode==='screen'){
      wa=0.3+0.4*lcWA+0.3*eWA;wb=0.3+0.4*lcWB+0.3*eWB;
    }else if(mode==='hsl'){
      wa=0.3+0.4*vWA+0.3*lcWA;wb=0.3+0.4*vWB+0.3*lcWB;
    }else{
      wa=0.34+0.33*eWA+0.33*vWA;wb=0.34+0.33*eWB+0.33*vWB;
    }

    // Coherence modulation:
    // When gradients conflict (low coherence), pull weights toward 50/50
    // to avoid one texture dominating and smearing the other
    const cohFactor=0.3*(1-coh); // max 30% pull toward equal
    wa=wa*(1-cohFactor)+0.5*cohFactor;
    wb=wb*(1-cohFactor)+0.5*cohFactor;

    // Structure protection:
    // Where both images have strong structure, preserve A more
    // (first uploaded image = primary subject intent)
    const structBoost=pA*0.15;
    wa+=structBoost;

    const t=wa+wb+1e-8;
    wA[i]=wa/t;
    wB[i]=wb/t;
  }

  // Spatial smoothing — removes weight map grid noise
  // sigma=0.8: tight enough to preserve local decisions
  const wAb=gaussBlur1ch(wA,w,h,0.8);
  const wBb=gaussBlur1ch(wB,w,h,0.8);
  for(let i=0;i<n;i++){const t=wAb[i]+wBb[i]+1e-8;wA[i]=wAb[i]/t;wB[i]=wBb[i]/t;}

  return{wA,wB};
}

// ── LAPLACIAN PYRAMID ─────────────────────────────────────────
const PYRAMID_LEVELS=6;
function downsample(data,w,h){
  const dw=Math.max(1,w>>1),dh=Math.max(1,h>>1),out=new Float32Array(dw*dh*3);
  for(let y=0;y<dh;y++){for(let x=0;x<dw;x++){
    let r=0,g=0,b=0,cnt=0;
    for(let dy=0;dy<2;dy++){for(let dx=0;dx<2;dx++){const sy=Math.min(h-1,y*2+dy),sx=Math.min(w-1,x*2+dx),si=(sy*w+sx)*3;r+=data[si];g+=data[si+1];b+=data[si+2];cnt++;}}
    const di=(y*dw+x)*3;out[di]=r/cnt;out[di+1]=g/cnt;out[di+2]=b/cnt;
  }}
  return{data:out,w:dw,h:dh};
}
function upsample(data,sw,sh,dw,dh){
  const out=new Float32Array(dw*dh*3);
  for(let y=0;y<dh;y++){for(let x=0;x<dw;x++){
    const fx=(x+0.5)*sw/dw-0.5,fy=(y+0.5)*sh/dh-0.5;
    const x0=Math.max(0,Math.min(sw-1,Math.floor(fx))),x1=Math.max(0,Math.min(sw-1,x0+1));
    const y0=Math.max(0,Math.min(sh-1,Math.floor(fy))),y1=Math.max(0,Math.min(sh-1,y0+1));
    const tx=fx-Math.floor(fx),ty=fy-Math.floor(fy),di=(y*dw+x)*3;
    for(let c=0;c<3;c++){
      const v00=data[(y0*sw+x0)*3+c],v10=data[(y0*sw+x1)*3+c],v01=data[(y1*sw+x0)*3+c],v11=data[(y1*sw+x1)*3+c];
      out[di+c]=(v00*(1-tx)+v10*tx)*(1-ty)+(v01*(1-tx)+v11*tx)*ty;
    }
  }}
  return out;
}
function buildPyramid(lin,w,h){
  const gauss=[{data:lin,w,h}];
  for(let lv=1;lv<PYRAMID_LEVELS;lv++){const prev=gauss[lv-1];gauss.push(downsample(gaussBlur(prev.data,prev.w,prev.h,1.0),prev.w,prev.h));}
  const pyr=[];
  for(let lv=0;lv<PYRAMID_LEVELS-1;lv++){
    const curr=gauss[lv],next=gauss[lv+1],up=upsample(next.data,next.w,next.h,curr.w,curr.h),lap=new Float32Array(curr.data.length);
    for(let i=0;i<curr.data.length;i++)lap[i]=curr.data[i]-up[i];
    pyr.push({lap,w:curr.w,h:curr.h});
  }
  const base=gauss[PYRAMID_LEVELS-1];pyr.push({lap:base.data,w:base.w,h:base.h});
  return pyr;
}
function collapsePyramid(pyr){
  let cur=pyr[PYRAMID_LEVELS-1].lap,cw=pyr[PYRAMID_LEVELS-1].w,ch=pyr[PYRAMID_LEVELS-1].h;
  for(let lv=PYRAMID_LEVELS-2;lv>=0;lv--){
    const up=upsample(cur,cw,ch,pyr[lv].w,pyr[lv].h),lap=pyr[lv].lap,out=new Float32Array(lap.length);
    for(let i=0;i<lap.length;i++)out[i]=up[i]+lap[i];
    cur=out;cw=pyr[lv].w;ch=pyr[lv].h;
  }
  return cur;
}

// ── PYRAMID FUSION ────────────────────────────────────────────
// Gaussian-weighted level resampling prevents seams between layers
function fusePyramids(pyrA,pyrB,wA,wB,w,h,mode,params,cohA,cohB){
  const fusedPyr=[];
  for(let lv=0;lv<PYRAMID_LEVELS;lv++){
    const levA=pyrA[lv],levB=pyrB[lv],lw=levA.w,lh=levA.h,ln=lw*lh;
    let levelWA,levelWB;
    if(lv===0){levelWA=wA;levelWB=wB;}
    else{
      levelWA=new Float32Array(ln);levelWB=new Float32Array(ln);
      const scX=w/lw,scY=h/lh;
      for(let y=0;y<lh;y++){for(let x=0;x<lw;x++){
        let sa=0,sb=0,wt=0;
        const x0=Math.floor(x*scX),x1=Math.min(w-1,Math.ceil((x+1)*scX));
        const y0=Math.floor(y*scY),y1=Math.min(h-1,Math.ceil((y+1)*scY));
        for(let sy=y0;sy<=y1;sy++){for(let sx=x0;sx<=x1;sx++){
          const dx=sx-x*scX,dy=sy-y*scY;
          const gw=Math.exp(-(dx*dx+dy*dy)/2.0);
          sa+=wA[sy*w+sx]*gw;sb+=wB[sy*w+sx]*gw;wt+=gw;
        }}
        levelWA[y*lw+x]=wt?sa/wt:0.5;levelWB[y*lw+x]=wt?sb/wt:0.5;
      }}
    }
    const dA=levA.lap,dB=levB.lap,fused=new Float32Array(lw*lh*3);
    if(lv<PYRAMID_LEVELS-1){
      // Fine/mid levels: weighted blend modulated by cross-scale coherence
      // Where coherence is low (noise/artifacts), reduce contribution
      // This prevents artifact amplification through the pyramid
      for(let i=0;i<ln;i++){
        const p=i*3,wa=levelWA[i],wb=levelWB[i];
        // Coherence at this level — downsample from full res
        // For levels > 0, coherence is already smoothed so simple lookup works
        const cA = cohA ? Math.min(1, (cohA[Math.min(cohA.length-1, i)] || 0) + 0.3) : 1;
        const cB = cohB ? Math.min(1, (cohB[Math.min(cohB.length-1, i)] || 0) + 0.3) : 1;
        // Suppress incoherent (noisy) frequencies, preserve coherent ones
        fused[p]  =dA[p]*wa*cA+dB[p]*wb*cB;
        fused[p+1]=dA[p+1]*wa*cA+dB[p+1]*wb*cB;
        fused[p+2]=dA[p+2]*wa*cA+dB[p+2]*wb*cB;
        // Renormalise by actual weight used
        const wt=(wa*cA+wb*cB)+1e-8;
        fused[p]/=wt;fused[p+1]/=wt;fused[p+2]/=wt;
      }
    }else{
      // Base level: mode-specific color interaction
      const alpha=params.alpha||0.5;
      for(let i=0;i<ln;i++){
        const p=i*3,wa=levelWA[i],wb=levelWB[i];
        const ra=dA[p],ga=dA[p+1],ba=dA[p+2],rb=dB[p],gb=dB[p+1],bb=dB[p+2];
        let r,g,b;
        if(mode==='multiply'){r=ra*rb;g=ga*gb;b=ba*bb;}
        else if(mode==='screen'){r=1-(1-ra)*(1-rb);g=1-(1-ga)*(1-gb);b=1-(1-ba)*(1-bb);}
        else if(mode==='overlay'){const ov=(a,bv)=>a<0.5?2*a*bv:1-2*(1-a)*(1-bv);r=ov(ra,rb);g=ov(ga,gb);b=ov(ba,bb);}
        else if(mode==='soft_light'){const sl=(a,bv)=>bv<0.5?a-(1-2*bv)*a*(1-a):a+(2*bv-1)*(Math.sqrt(Math.max(0,a))-a);r=sl(ra,rb);g=sl(ga,gb);b=sl(ba,bb);}
        else if(mode==='difference'){r=Math.abs(ra-rb);g=Math.abs(ga-gb);b=Math.abs(ba-bb);}
        else if(mode==='normal'){r=ra*(1-alpha)+rb*alpha;g=ga*(1-alpha)+gb*alpha;b=ba*(1-alpha)+bb*alpha;}
        else{r=ra*wa+rb*wb;g=ga*wa+gb*wb;b=ba*wa+bb*wb;}
        // Blend mode result with adaptive weighted average
        // 60/40 — mode gives character, adaptive prevents color poisoning
        fused[p]  =r*0.6+(ra*wa+rb*wb)*0.4;
        fused[p+1]=g*0.6+(ga*wa+gb*wb)*0.4;
        fused[p+2]=b*0.6+(ba*wa+bb*wb)*0.4;
      }
    }
    fusedPyr.push({lap:fused,w:lw,h:lh});
  }
  return fusedPyr;
}

// ── ACES TONE MAPPING ─────────────────────────────────────────
// Applied after pyramid collapse. Prevents blown highlights and glow.
function toneMapACES(x){const a=2.51,b=0.03,c=2.43,d=0.59,e2=0.14;return Math.max(0,Math.min(1,(x*(a*x+b))/(x*(c*x+d)+e2)));}
function applyACES(data){const out=new Float32Array(data.length);for(let i=0;i<data.length;i++)out[i]=toneMapACES(data[i]);return out;}

// ── LOW-FREQUENCY RESTORATION ─────────────────────────────────
// Prevents high-freq fusion from destroying global scene structure.
// sigma=3.0 captures broad lighting, sky gradients, reflections.
function lowFreqRestore(output,linA,linB,w,h){
  const lowA=gaussBlur(linA,w,h,3.0),lowB=gaussBlur(linB,w,h,3.0);
  const out=new Float32Array(output.length);
  for(let i=0;i<output.length;i++){
    const base=(lowA[i]+lowB[i])*0.5;
    out[i]=output[i]*0.7+base*0.3;
  }
  return out;
}

// ── OKLAB CHROMA CORRECTION ───────────────────────────────────
// Replaces HSL pass. OKLab interpolation is perceptually uniform:
// no neon shifts, no saturation blooms, smooth color transitions.
// amount: 0=pure pyramid result, 1=full OKLab blend
// Saturation clamped at 0.85 to prevent oversaturation.
function oklabChromaPass(fused,linA,linB,wA,wB,n,amount){
  if(amount<=0)return fused;
  const out=new Float32Array(n*3);
  for(let i=0;i<n;i++){
    const p=i*3;
    // Get OKLab of fused — L (lightness) from pyramid stays
    const[Lf,af,bf]=linearToOklab(fused[p],fused[p+1],fused[p+2]);
    // Get OKLab of sources
    const[La,aa,ba]=linearToOklab(linA[p],linA[p+1],linA[p+2]);
    const[Lb,ab,bb]=linearToOklab(linB[p],linB[p+1],linB[p+2]);
    // Blend a,b chroma channels weighted by adaptive maps
    const wa=wA[i],wb=wB[i];
    const aBlend=aa*wa+ab*wb;
    const bBlend=ba*wa+bb*wb;
    // Clamp chroma magnitude to prevent oversaturation
    const chromaMag=Math.sqrt(aBlend*aBlend+bBlend*bBlend);
    const maxChroma=0.32; // OKLab chroma limit for natural images
    const scale=chromaMag>maxChroma?maxChroma/chromaMag:1;
    const aFinal=af*(1-amount)+(aBlend*scale)*amount;
    const bFinal=bf*(1-amount)+(bBlend*scale)*amount;
    // L from pyramid is authoritative — chroma correction only
    const[r,g,b]=oklabToLinear(Lf,aFinal,bFinal);
    // Defer clamping to final encode
    out[p]=r;out[p+1]=g;out[p+2]=b;
  }
  return out;
}

// ── EDGE-GATED UNSHARP MASK ───────────────────────────────────
// Sharpens detail in flat/texture regions.
// Suppresses at strong edges (already sharp) and noisy areas.
function edgeAwareUnsharp(lin,edgeMapA,edgeMapB,w,h,amount){
  if(amount<=0)return lin;
  const blurred=gaussBlur(lin,w,h,1.2);
  const out=new Float32Array(lin.length);
  const n=w*h;
  for(let i=0;i<n;i++){
    const edge=(edgeMapA[i]+edgeMapB[i])*0.5;
    // Suppress at edges (edge>0.4), boost in mid-texture
    const strength=amount*(1.0-edge*0.85);
    const p=i*3;
    out[p]  =lin[p]  +strength*(lin[p]  -blurred[p]);
    out[p+1]=lin[p+1]+strength*(lin[p+1]-blurred[p+1]);
    out[p+2]=lin[p+2]+strength*(lin[p+2]-blurred[p+2]);
  }
  return out;
}

// ── DITHERING ─────────────────────────────────────────────────
// Breaks up banding in gradients and skies.
// Applied just before sRGB encode, after all processing.
function dither(lin){
  const out=new Float32Array(lin.length);
  for(let i=0;i<lin.length;i++)out[i]=lin[i]+(Math.random()-0.5)*0.003;
  return out;
}

// ── DIRECT BLEND ──────────────────────────────────────────────
function blendDirect(linA,linB,wA,wB,n,mode,params){
  const out=new Float32Array(n*3),atA=new Float32Array(n),atB=new Float32Array(n),alpha=params.alpha||0.5;
  for(let i=0;i<n;i++){
    const p=i*3,wa=wA[i],wb=wB[i];
    const ra=linA[p],ga=linA[p+1],ba=linA[p+2],rb=linB[p],gb=linB[p+1],bb=linB[p+2];
    let r,g,b;
    if(mode==='normal'){r=ra*(1-alpha)+rb*alpha;g=ga*(1-alpha)+gb*alpha;b=ba*(1-alpha)+bb*alpha;}
    else if(mode==='multiply'){r=ra*rb;g=ga*gb;b=ba*bb;}
    else if(mode==='difference'){r=Math.abs(ra-rb);g=Math.abs(ga-gb);b=Math.abs(ba-bb);}
    else if(mode==='detail'){r=rb*wb+ra*wa;g=gb*wb+ga*wa;b=bb*wb+ba*wa;}
    else{r=ra*wa+rb*wb;g=ga*wa+gb*wb;b=ba*wa+bb*wb;}
    out[p]=r*0.65+(ra*wa+rb*wb)*0.35;out[p+1]=g*0.65+(ga*wa+gb*wb)*0.35;out[p+2]=b*0.65+(ba*wa+bb*wb)*0.35;
    atA[i]=wa;atB[i]=wb;
  }
  return{out,atA,atB};
}

// ── MAIN HANDLER ──────────────────────────────────────────────
self.onmessage=function(e){
  const{rgbaA,rgbaB,sw,sh,dw,dh,mode,params,unsharpAmount}=e.data;
  const n=dw*dh;

  self.postMessage({type:'progress',pct:4, text:'resampling A\u2026'});
  const resA=lanczosResize(rgbaA,sw,sh,dw,dh);
  self.postMessage({type:'progress',pct:13,text:'resampling B\u2026'});
  const resB=lanczosResize(rgbaB,sw,sh,dw,dh);

  self.postMessage({type:'progress',pct:22,text:'linearising\u2026'});
  const linA=toLinearBuf(resA,n);
  const linB=toLinearBuf(resB,n);

  self.postMessage({type:'progress',pct:30,text:'analysing scene A\u2026'});
  const aA=analyseScene(linA,dw,dh);
  self.postMessage({type:'progress',pct:38,text:'analysing scene B\u2026'});
  const aB=analyseScene(linB,dw,dh);

  self.postMessage({type:'progress',pct:45,text:'computing weights\u2026'});
  const{wA,wB}=computeWeightMaps(aA,aB,dw,dh,mode,params);

  const usePyramid=['frequency','hsl','soft_light','overlay','screen'].includes(mode);
  let fused,atA,atB;

  if(usePyramid){
    self.postMessage({type:'progress',pct:50,text:'bilateral pre-filter\u2026'});
    // Bilateral filter removes JPEG/compression artifacts before pyramid
    // decomposition. sigmaS=1.5 (spatial), sigmaR=0.06 (range in linear)
    // Light touch — just enough to remove artifact amplification
    const filtA = bilateralFilter(linA, dw, dh, 1.5, 0.06);
    const filtB = bilateralFilter(linB, dw, dh, 1.5, 0.06);

    self.postMessage({type:'progress',pct:56,text:'building pyramids\u2026'});
    const pyrA=buildPyramid(filtA,dw,dh);
    const pyrB=buildPyramid(filtB,dw,dh);

    self.postMessage({type:'progress',pct:66,text:'cross-scale coherence\u2026'});
    // Identify real detail vs noise using pyramid persistence
    const cohA = crossScaleCoherence(pyrA, dw, dh);
    const cohB = crossScaleCoherence(pyrB, dw, dh);

    self.postMessage({type:'progress',pct:72,text:'fusing pyramids\u2026'});
    const fp=fusePyramids(pyrA,pyrB,wA,wB,dw,dh,mode,params,cohA,cohB);

    self.postMessage({type:'progress',pct:74,text:'collapsing\u2026'});
    let collapsed=collapsePyramid(fp);

    self.postMessage({type:'progress',pct:79,text:'tone mapping\u2026'});
    collapsed=applyACES(collapsed);

    self.postMessage({type:'progress',pct:83,text:'restoring structure\u2026'});
    collapsed=lowFreqRestore(collapsed,linA,linB,dw,dh);

    self.postMessage({type:'progress',pct:86,text:'OKLab colour correction\u2026'});
    // Gentle: 0.08 for photographic, 0.35 for hsl mode
    const chromaAmt=mode==='hsl'?0.35:0.08;
    fused=oklabChromaPass(collapsed,linA,linB,wA,wB,n,chromaAmt);
    atA=wA;atB=wB;
  }else{
    self.postMessage({type:'progress',pct:55,text:'blending\u2026'});
    const result=blendDirect(linA,linB,wA,wB,n,mode,params);
    fused=result.out;atA=result.atA;atB=result.atB;
    self.postMessage({type:'progress',pct:74,text:'colour harmonisation\u2026'});
    fused=oklabChromaPass(fused,linA,linB,wA,wB,n,0.04);
  }

  self.postMessage({type:'progress',pct:89,text:'sharpening\u2026'});
  const sharpened=unsharpAmount>0
    ?edgeAwareUnsharp(fused,aA.edgeMap,aB.edgeMap,dw,dh,unsharpAmount*0.35)
    :fused;

  self.postMessage({type:'progress',pct:93,text:'dithering\u2026'});
  const dithered=dither(sharpened);

  self.postMessage({type:'progress',pct:96,text:'encoding\u2026'});
  const rgba=toSRGBBuf(dithered,n);
  const rgbaAout=toSRGBBuf(linA,n);
  const rgbaBout=toSRGBBuf(linB,n);

  self.postMessage({type:'done',rgba,rgbaAout,rgbaBout,atA,atB,dw,dh},
    [rgba.buffer,rgbaAout.buffer,rgbaBout.buffer,atA.buffer,atB.buffer]);
};
