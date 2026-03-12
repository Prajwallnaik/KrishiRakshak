import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000/predict'

function formatClass(raw = '') {
  return raw.replace(/^Tomato___/, '').replaceAll('_', ' ')
}

/* ══════════════════════════════════════════════
   PREMIUM INTERACTIONS
   ══════════════════════════════════════════════ */

/* ── Custom Interactive Cursor ── */
function CustomCursor() {
  const dotRef = useRef(null)
  const ringRef = useRef(null)

  useEffect(() => {
    const onMouseMove = (e) => {
      if (dotRef.current) {
        dotRef.current.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`
      }
      if (ringRef.current) {
        // slight delay for the ring using a simple lerp in rAF would be better, but direct translation works for now
        ringRef.current.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`
      }
    }

    // add hover states for clickable elements
    const onMouseOver = (e) => {
      const isClickable = e.target.closest('a, button, input, .magnetic, .dropzone, .tech-card')
      if (ringRef.current) {
        if (isClickable) {
          ringRef.current.classList.add('cursor-hover')
        } else {
          ringRef.current.classList.remove('cursor-hover')
        }
      }
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseover', onMouseOver)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseover', onMouseOver)
    }
  }, [])

  return (
    <>
      <div ref={ringRef} className="cursor-ring" />
      <div ref={dotRef} className="cursor-dot" />
    </>
  )
}

/* ── Magnetic Wrapper ── */
function Magnetic({ children, className = '', onClick }) {
  const ref = useRef(null)

  const handleMouseMove = (e) => {
    const el = ref.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = e.clientX - rect.left - rect.width / 2
    const y = e.clientY - rect.top - rect.height / 2
    el.style.transform = `translate(${x * 0.15}px, ${y * 0.15}px)`
  }

  const handleMouseLeave = () => {
    if (ref.current) {
      ref.current.style.transform = 'translate(0px, 0px)'
    }
  }

  return (
    <div
      ref={ref}
      className={`magnetic ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      style={{ transition: 'transform 0.2s cubic-bezier(0.2, 0.8, 0.2, 1)', display: 'inline-block' }}
    >
      {children}
    </div>
  )
}

/* ── 3D Tilt Card Wrapper ── */
function TiltCard({ children, className = '', style: externalStyle }) {
  const ref = useRef(null)

  const handleMouseMove = (e) => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const centerX = rect.width / 2
    const centerY = rect.height / 2
    const rotateX = ((y - centerY) / centerY) * -5
    const rotateY = ((x - centerX) / centerX) * 5
    ref.current.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`
  }

  const handleMouseLeave = () => {
    if (ref.current) {
      ref.current.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)'
    }
  }

  return (
    <div
      ref={ref}
      className={`tilt-card ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ transition: 'transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1)', willChange: 'transform', ...externalStyle }}
    >
      {children}
    </div>
  )
}

/* ══════════════════════════════════════════════
   CINEMATIC SCROLL HOOKS
   ══════════════════════════════════════════════ */

/* ── Global scroll Y tracker (rAF-driven, silky smooth) ── */
function useScrollY() {
  const [y, setY] = useState(0)
  useEffect(() => {
    let ticking = false
    const onScroll = () => {
      if (!ticking) {
        ticking = true
        requestAnimationFrame(() => {
          setY(window.scrollY)
          ticking = false
        })
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])
  return y
}

/* ── Section progress: 0 when section enters bottom, 1 when it exits top ── */
function useSectionProgress(ref) {
  const [progress, setProgress] = useState(0)
  useEffect(() => {
    let ticking = false
    const update = () => {
      if (!ref.current) return
      const rect = ref.current.getBoundingClientRect()
      const vh = window.innerHeight
      const p = Math.min(Math.max((vh - rect.top) / (vh + rect.height), 0), 1)
      setProgress(p)
      ticking = false
    }
    const onScroll = () => {
      if (!ticking) { ticking = true; requestAnimationFrame(update) }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    update()
    return () => window.removeEventListener('scroll', onScroll)
  }, [ref])
  return progress
}

/* ── Cinematic scroll reveal (directional variants with stagger) ── */
function useCinematicReveal() {
  useEffect(() => {
    const obs = new IntersectionObserver(
      entries => entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible')
          // Stagger children: look on self or nested [data-stagger]
          const staggerRoot = e.target.dataset.stagger !== undefined
            ? e.target
            : e.target.querySelector('[data-stagger]')
          if (staggerRoot) {
            const children = staggerRoot.querySelectorAll('.reveal-child')
            children.forEach((child, i) => {
              child.style.transitionDelay = `${i * 90}ms`
              requestAnimationFrame(() => child.classList.add('visible'))
            })
          }
        }
      }),
      { threshold: 0.06, rootMargin: '0px 0px -30px 0px' }
    )
    document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale, .reveal-blur, .reveal-tilt, .reveal-child').forEach(el => obs.observe(el))
    return () => obs.disconnect()
  })
}

/* ── Canvas starfield with parallax ── */
function Starfield({ scrollY }) {
  const ref = useRef()
  const starsRef = useRef(null)

  useEffect(() => {
    const c = ref.current, ctx = c.getContext('2d')
    let W = c.width = window.innerWidth, H = c.height = window.innerHeight
    const stars = Array.from({ length: 220 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      r: Math.random() * 1.3 + 0.2,
      t: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.5 + 0.2,
      col: ['#2563EB', '#60A5FA', '#94A3B8', '#CBD5E1', '#3B82F6'][~~(Math.random() * 5)]
    }))
    starsRef.current = stars
    let raf, prev = 0
    const draw = (ts) => {
      const dt = Math.min((ts - prev) / 1000, 0.05)
      prev = ts
      ctx.clearRect(0, 0, W, H)
      stars.forEach(s => {
        s.t += dt * s.speed
        ctx.globalAlpha = 0.25 + 0.55 * (Math.sin(s.t) * .5 + .5)
        ctx.fillStyle = s.col
        ctx.beginPath()
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2)
        ctx.fill()
      })
      ctx.globalAlpha = 1
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    const resize = () => { W = c.width = window.innerWidth; H = c.height = window.innerHeight }
    window.addEventListener('resize', resize)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])

  const parallaxOffset = scrollY * 0.15
  return (
    <canvas
      id="starfield"
      ref={ref}
      style={{ transform: `translateY(${parallaxOffset}px)` }}
    />
  )
}

/* ── Animated number counter for stats ── */
function CountUp({ target, suffix = '' }) {
  const [val, setVal] = useState(0)
  const ref = useRef()
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return
      obs.disconnect()
      const num = parseFloat(target)
      const isFloat = target.includes('.')
      const dur = 1400
      const start = performance.now()
      const tick = (now) => {
        const p = Math.min((now - start) / dur, 1)
        const ease = 1 - Math.pow(1 - p, 3)
        setVal(isFloat ? (num * ease).toFixed(1) : Math.round(num * ease))
        if (p < 1) requestAnimationFrame(tick)
        else setVal(isFloat ? num.toFixed(1) : num)
      }
      requestAnimationFrame(tick)
    }, { threshold: 0.5 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [target])
  return <span ref={ref}>{val}{suffix}</span>
}

/* ── Horizontal marquee ticker ── */
function Marquee() {
  const items = [
    'Smart Plant Care', '10 Plant Conditions', 'Highly Accurate',
    'Quick Detection', 'Simple Explanations', 'Real-time Diagnosis',
    'Farmer Friendly', 'Instant Results', 'KrishiRakshak AI'
  ]
  const track = items.concat(items).map((text, i) => (
    <span className="marquee-item" key={i}>
      {text}
      <span className="marquee-dot">✦</span>
    </span>
  ))
  return (
    <div className="marquee-strip">
      <div className="marquee-track">{track}</div>
    </div>
  )
}

const DISEASES = [
  'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
  'Septoria Leaf Spot', 'Spider Mites', 'Target Spot',
  'Tomato Mosaic Virus', 'Yellow Leaf Curl Virus', 'Healthy',
]

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [recLoading, setRecLoading] = useState(false)
  const [activeNav, setActiveNav] = useState('analyse')
  const [result, setResult] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const scrollY = useScrollY()
  const heroRef = useRef()
  useCinematicReveal()

  /* ── Navbar scroll state ── */
  const navScrolled = scrollY > 80
  const scrollPercent = Math.min(scrollY / (document.documentElement.scrollHeight - window.innerHeight || 1), 1)


  const fetchRecommendations = async (disease) => {
    if (disease.toLowerCase().includes('healthy')) {
      setRecommendations({
        symptoms: "The leaf appears healthy and vibrant.",
        causes: "Proper care and favorable conditions.",
        organic: "Continue regular composting, mulching, and balanced watering.",
        chemical: "No chemical intervention needed."
      })
      return
    }

    setRecLoading(true)
    try {
      // Hit our new backend endpoint which securely calls Gemini
      const res = await fetch(`http://localhost:8000/recommend/${encodeURIComponent(disease)}`)
      if (!res.ok) throw new Error('Failed to fetch recommendations from backend')

      const data = await res.json()
      setRecommendations(data)
    } catch (e) {
      console.error('LLM Error:', e)
      setRecommendations({
        symptoms: "Error fetching live recommendations.",
        causes: "Connection issue with the local backend.",
        organic: "Please consult a local agricultural expert.",
        chemical: "Check pesticide labels for target diseases."
      })
    } finally {
      setRecLoading(false)
    }
  }

  const pick = useCallback((f) => {
    if (!f) return
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(f.type)) {
      setError('Please upload a JPEG or PNG image.'); return
    }
    setFile(f); setPreview(URL.createObjectURL(f))
    setResult(null); setRecommendations(null); setError(null)
  }, [])

  const onDrop = e => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]) }

  const analyse = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null); setRecommendations(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await fetch(API_URL, { method: 'POST', body: fd })
      if (!res.ok) { const j = await res.json(); throw new Error(j.detail || 'Server error') }
      const data = await res.json()
      setResult(data)
      setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 80)
    } catch (e) {
      setError(e.message || 'Could not reach the API — make sure the server is on port 8000.')
    } finally { setLoading(false) }
  }

  const topProbs = result
    ? Object.entries(result.probabilities || {})
      .map(([k, v]) => [formatClass(k), parseFloat(v)])
      .sort((a, b) => b[1] - a[1]).slice(0, 5)
    : []

  const scrollTo = id => e => {
    e?.preventDefault();
    setActiveNav(id);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <>
      <CustomCursor />
      {/* ── NAVBAR — Premium Liquid Glass ── */}
      <div style={{ position: 'fixed', top: 24, width: '100%', display: 'flex', justifyContent: 'center', zIndex: 200 }}>
        <nav className="glass-nav">
          {/* Brand Logo */}
          <div className="nav-brand">
            <img
              src="https://freepnglogo.com/images/all_img/1723808808meta-logo-transparent-PNG.png"
              alt="KrishiRakshak logo"
              className="nav-logo-img"
            />
            <span className="nav-brand-name">KrishiRakshak</span>
          </div>
          <div className="nav-divider" />
          <ul className="nav-list">
            <li><a href="#about" className={`nav-item ${activeNav === 'about' ? 'active' : ''}`} onClick={scrollTo('about')}>About</a></li>
            <li><a href="#analyse" className={`nav-item ${activeNav === 'analyse' ? 'active' : ''}`} onClick={scrollTo('analyse')}>Analyse</a></li>
            {result && <li><a href="#results" className={`nav-item ${activeNav === 'results' ? 'active' : ''}`} onClick={scrollTo('results')}>Results</a></li>}
          </ul>
        </nav>
      </div >

      {/* ── HERO with parallax depth ── */}
      <section className="hero" ref={heroRef} >
        <Starfield scrollY={scrollY} />
        <div className="hero-content">

          <div className="ambient-glow" />
          <h1 className="hero-title-anim" style={{ transform: `translateY(${scrollY * 0.12}px)` }}>
            <span className="word w1">Detect</span> <span className="word w2">tomato</span><br />
            <span className="grad word w3">plant</span> <span className="grad word w4">diseases</span><br />
            <span className="word w5">instantly.</span>
          </h1>
          <p className="hero-sub reveal-blur" style={{ transform: `translateY(${scrollY * 0.18}px)` }}>
            One photograph. Ten conditions. Instant answers.<br />
            Upload a leaf photo and our system instantly identifies the problem, helping you protect your harvest.
          </p>
          <div className="hero-actions" style={{ transform: `translateY(${scrollY * 0.22}px)` }}>
            <Magnetic>
              <button className="btn-hero solid" onClick={scrollTo('analyse')}>
                Analyse a Leaf →
              </button>
            </Magnetic>
            <Magnetic>
              <button className="btn-hero ghost" onClick={scrollTo('about')}>
                Learn more
              </button>
            </Magnetic>
          </div>
        </div>
        <div className="scroll-cue" style={{ opacity: Math.max(1 - scrollY / 200, 0) }}>
          <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12l7 7 7-7" />
          </svg>
          Scroll
        </div>
      </section >

      {/* ── MARQUEE TICKER ── */}
      <Marquee />

      {/* ── ABOUT SECTION with cinematic reveals ── */}
      <section className="about-section" id="about" >
        <div className="about-inner">
          <div className="about-header">
            <div className="section-eyebrow reveal-blur" style={{ color: '#c97b3a' }}>About KrishiRakshak AI</div>
            <h2 className="about-title reveal-blur">
              What is KrishiRakshak,<br />
              <span className="grad">and what can it do?</span>
            </h2>
            <p className="about-sub reveal-blur">
              KrishiRakshak AI is a smart plant disease detection tool. Upload a photo of a
              tomato leaf and the app instantly tells you whether the plant is healthy or
              identifies the problem — helping farmers and gardeners act faster and more accurately.
            </p>
          </div>

          {/* Feature cards — staggered scale reveal */}
          <div className="tech-grid reveal-scale" data-stagger="">
            {[
              {
                svg: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="20" height="20">
                    <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
                  </svg>
                ),
                color: '#c97b3a', bg: 'rgba(210,140,60,.12)', border: 'rgba(210,140,60,.22)',
                label: 'Fast Results',
                desc: 'Receive a precise diagnosis within seconds of uploading — no expertise or lab required.',
              },
              {
                svg: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="20" height="20">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                ),
                color: '#b07040', bg: 'rgba(190,120,60,.10)', border: 'rgba(190,120,60,.20)',
                label: 'Wide Coverage',
                desc: 'Detects 10 distinct tomato leaf conditions — from early blight to mosaic virus — in one analysis.',
              },
              {
                svg: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="20" height="20">
                    <path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                  </svg>
                ),
                color: '#d4863a', bg: 'rgba(212,134,58,.12)', border: 'rgba(212,134,58,.22)',
                label: 'Treatment Plan',
                desc: 'Get clear, step-by-step guidance on both organic and chemical treatment options for your crop.',
              },
              {
                svg: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="20" height="20">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                  </svg>
                ),
                color: '#6aab5e', bg: 'rgba(100,180,90,.10)', border: 'rgba(100,180,90,.22)',
                label: 'Confidence Score',
                desc: 'Every diagnosis includes a reliability percentage so you know exactly how trustworthy the result is.',
              },
            ].map(({ svg, color, bg, border, label, desc }) => (
              <TiltCard className="tech-card reveal-child" key={label}>
                <div className="tech-card-content">
                  <div className="tech-icon" style={{ color, background: bg, border: `1px solid ${border}` }}>
                    {svg}
                  </div>
                  <div className="tech-label">{label}</div>
                  <div className="tech-desc">{desc}</div>
                </div>
                <div className="tech-card-glow" style={{ background: bg }} />
              </TiltCard>
            ))}
          </div>

          {/* How to use — redesigned with icon badges */}
          <div className="pipeline reveal-left">
            <div className="pipeline-header">
              <div className="pipeline-eyebrow">How to use KrishiRakshak AI</div>
            </div>
            <div className="pipeline-steps-v2">
              {[
                {
                  num: '01',
                  icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18"><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" /></svg>,
                  title: 'Photograph your leaf',
                  body: 'Take a clear, well-lit photo of a single tomato leaf — JPEG or PNG format.',
                },
                {
                  num: '02',
                  icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>,
                  title: 'Upload your image',
                  body: 'Drag the photo into the upload zone, or simply click to browse from your device.',
                },
                {
                  num: '03',
                  icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>,
                  title: 'Run the analysis',
                  body: 'Hit Analyse — our model identifies the condition from 10 possible diagnoses in seconds.',
                },
                {
                  num: '04',
                  icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /><polyline points="10 9 9 9 8 9" /></svg>,
                  title: 'Review your treatment plan',
                  body: 'Receive a full report with the disease name, confidence score, and recommended action steps.',
                },
              ].map(({ num, icon, title, body }, idx, arr) => (
                <div className="pipeline-step-v2 reveal-child" key={num}>
                  <div className="psv2-left">
                    <div className="psv2-badge">
                      {icon}
                    </div>
                    {idx < arr.length - 1 && <div className="psv2-connector" />}
                  </div>
                  <div className="psv2-body">
                    <div className="psv2-num">{num}</div>
                    <div className="psv2-title">{title}</div>
                    <div className="psv2-desc">{body}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detectable diseases — upgraded visual tags */}
          <div className="disease-list reveal-right" data-stagger="">
            <div className="disease-header">
              <div className="pipeline-eyebrow">Detectable Conditions</div>
              <div className="disease-count-badge">10 conditions</div>
            </div>
            <div className="disease-tags">
              {[
                { name: 'Bacterial Spot', dot: '#e05a3a' },
                { name: 'Early Blight', dot: '#c97b3a' },
                { name: 'Late Blight', dot: '#b85c30' },
                { name: 'Leaf Mold', dot: '#a67c52' },
                { name: 'Septoria Leaf Spot', dot: '#d4863a' },
                { name: 'Spider Mites', dot: '#c97b3a' },
                { name: 'Target Spot', dot: '#b07040' },
                { name: 'Tomato Mosaic Virus', dot: '#e05a3a' },
                { name: 'Yellow Leaf Curl Virus', dot: '#d4a03a' },
                { name: 'Healthy', dot: '#6aab5e' },
              ].map(({ name, dot }) => (
                <span className="disease-tag reveal-child" key={name}>
                  <span className="disease-dot" style={{ background: dot }} />
                  {name}
                </span>
              ))}
            </div>
          </div>

        </div>{/* end about-inner */}
      </section >


      {/* ── STATS BAND REMOVED ── */}

      {/* ── UPLOAD SECTION with perspective tilt ── */}
      <section className="upload-section" id="analyse" >
        <div className="section-inner">
          <div className="section-eyebrow-lt reveal-blur">Disease Detection</div>
          <h2 className="section-title reveal-blur">
            Drop your leaf.<br />Get your diagnosis.
          </h2>
          <p className="section-sub reveal-blur">
            Our system has learned from thousands of tomato leaf photos
            to identify 10 different conditions. Simply upload and let it work.
          </p>

          <div className="upload-card reveal-tilt">
            <div
              className={`dropzone${dragging ? ' dragging' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input ref={inputRef} type="file" accept="image/jpeg,image/png"
                style={{ display: 'none' }} onChange={e => pick(e.target.files[0])} />
              <div className={`dz-ring ${preview ? 'success-pulse' : ''}`}>
                {preview
                  ? <svg viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" width="30" height="30"><path d="M20 6L9 17l-5-5" /></svg>
                  : <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="30" height="30"><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="M21 15l-5-5L5 21" /></svg>
                }
              </div>
              {preview
                ? <><div className="dz-filename" title={file?.name}>{file?.name}</div>
                  <div className="dz-hint-text">Click to change image</div></>
                : <><div className="dz-primary">
                  <span>Click to upload</span> or drag &amp; drop
                </div>
                  <div className="dz-hint-text">JPEG or PNG · Tomato leaves only</div></>
              }
            </div>

            {preview && (
              <div className="preview">
                <img src={preview} alt="Leaf preview" className="preview-image-anim" />
              </div>
            )}

            <hr className="upload-divider" />

            <div className="upload-footer">
              <div className="upload-meta" title={file?.name}>
                {file
                  ? <><strong>{file.name}</strong> · {(file.size / 1024).toFixed(1)} KB</>
                  : 'No file selected'}
              </div>
              {file && (
                <button className="btn-analyse-inline" onClick={analyse} disabled={loading}>
                  {loading
                    ? <><span className="spinner" />Analysing…</>
                    : 'Analyse'}
                </button>
              )}
            </div>

            {error && <div className="error-box">{error}</div>}
          </div>
        </div>
      </section >

      {/* ── RESULTS ── */}
      {
        result && (
          <section className="result-section" id="results">
            <div className="result-inner">
              <div className="result-header reveal-blur">
                <div className="result-tag">Diagnosis</div>
                <div className="result-class-name">{formatClass(result.predicted_class)}</div>
                <div className="result-conf">Certainty: {result.confidence}</div>
              </div>

              <div className="result-cards" data-stagger="">
                <TiltCard className="rcard reveal-scale reveal-child">
                  <div className="rcard-label">Predicted Disease</div>
                  <div className="rcard-val">{formatClass(result.predicted_class)}</div>
                </TiltCard>
                <TiltCard className="rcard reveal-scale reveal-child" style={{ transitionDelay: '100ms' }}>
                  <div className="rcard-label">Certainty Score</div>
                  <div className="rcard-val green">{result.confidence}</div>
                </TiltCard>
              </div>

              {/* LLM Recommendations Section */}
              <div className="llm-section reveal-blur">
                <div className="llm-header">
                  <div className="llm-badge">Smart Assistant</div>
                  <h3 className="llm-title">Treatment Recommendations</h3>
                </div>

                {!recommendations && !recLoading ? (
                  <div className="llm-prompt">
                    <button
                      className="btn-analyse-inline"
                      onClick={() => fetchRecommendations(formatClass(result.predicted_class))}
                    >
                      Get Treatment Recommendations
                    </button>
                  </div>
                ) : recLoading ? (
                  <div className="llm-loading">
                    <div className="spinner" style={{ width: 24, height: 24, borderTopColor: 'var(--accent)' }} />
                    <span>Consulting plant expert...</span>
                  </div>
                ) : recommendations ? (
                  <div className="llm-grid" data-stagger="">
                    <div className="llm-card reveal-child">
                      <div className="llm-icon-wrapper symptoms">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                      </div>
                      <div className="llm-card-tag">Symptoms</div>
                      <p>{recommendations.symptoms}</p>
                    </div>
                    <div className="llm-card reveal-child">
                      <div className="llm-icon-wrapper causes">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>
                      </div>
                      <div className="llm-card-tag">Causes</div>
                      <p>{recommendations.causes}</p>
                    </div>
                    <div className="llm-card reveal-child">
                      <div className="llm-icon-wrapper organic">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 8h-3c-1 0-3 2-3 2a7 7 0 0 1-3 8z" /><path d="M11 20a7 7 0 0 1-5-2.1c1.2-1.9 2-3.4 1-5.1" /><path d="M11 13a7 7 0 0 0 2-4.1" /></svg>
                      </div>
                      <div className="llm-card-tag green">Organic Treatment</div>
                      <p>{recommendations.organic}</p>
                    </div>
                    <div className="llm-card reveal-child">
                      <div className="llm-icon-wrapper chemical">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" /></svg>
                      </div>
                      <div className="llm-card-tag rose">Chemical Treatment</div>
                      <p>{recommendations.chemical}</p>
                    </div>
                  </div>
                ) : null}
              </div>



              <div className="result-cta reveal-scale">
                <Magnetic>
                  <button className="btn-reset" onClick={() => { setFile(null); setPreview(null); setResult(null); setRecommendations(null); scrollTo('analyse')() }}>
                    Analyse another leaf
                  </button>
                </Magnetic>
              </div>
            </div>
          </section>
        )
      }

      {/* ── FOOTER ── */}
      <footer className="site-footer reveal">
        <div className="footer-main">
          <div className="footer-brand">
            <div className="footer-brand-name">
              <img
                src="https://freepnglogo.com/images/all_img/1723808808meta-logo-transparent-PNG.png"
                alt="KrishiRakshak logo"
                className="nav-logo-img"
              />
              KrishiRakshak AI
            </div>
            <p>A smart, easy-to-use tool for tomato leaf disease detection, providing instant insights and advice for farmers and gardeners.</p>
          </div>
          <div className="footer-links-group-wrapper">
            <div className="footer-links-group">
              <h4>Resources</h4>
              <span className="non-link">Disease Guide</span>
              <span className="non-link">Treatment API</span>
              <span className="non-link">Farming Blog</span>
            </div>
            <div className="footer-links-group">
              <h4>Company</h4>
              <span className="non-link">About Us</span>
              <span className="non-link">Careers</span>
              <span className="non-link">Contact</span>
            </div>
            <div className="footer-links-group">
              <h4>Legal</h4>
              <span className="non-link">Privacy Policy</span>
              <span className="non-link">Terms of Service</span>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <span>© 2026 KrishiRakshak AI. All rights reserved.</span>
          <span>Smart AI · Fast · Reliable · Secure</span>
        </div>
      </footer>
    </>
  )
}
