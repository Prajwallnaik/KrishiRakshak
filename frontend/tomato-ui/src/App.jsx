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
    'Farmer Friendly', 'Instant Results', 'Samatva Krishi AI'
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

  const triggerHaptic = () => {
    if (typeof window !== 'undefined' && window.navigator && window.navigator.vibrate) {
      window.navigator.vibrate(50);
    }
  };

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
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 35000)
      const res = await fetch(
        `http://localhost:8000/recommend/${encodeURIComponent(disease)}`,
        { signal: controller.signal }
      )
      clearTimeout(timer)
      if (!res.ok) throw new Error('Failed to fetch recommendations from backend')
      const data = await res.json()
      setRecommendations(data)
    } catch (e) {
      console.error('LLM Error:', e)
      setRecommendations({
        symptoms: "Could not load live data — showing offline advice.",
        causes: "Connection issue or timeout with the recommendation service.",
        organic: "Please consult a local agricultural expert for organic options.",
        chemical: "Check pesticide labels for the detected disease."
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
      // Auto-fetch AI recommendations right after diagnosis
      fetchRecommendations(formatClass(data.predicted_class))
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
      {/* ── NAVBAR — Premium Liquid Glass ── */}
      <div style={{ position: 'fixed', top: 24, width: '100%', display: 'flex', justifyContent: 'center', zIndex: 200 }}>
        <nav className="glass-nav">
          {/* Brand Logo */}
          <div className="nav-brand">
            <img
              src="/logo.jpeg"
              alt="Samatva Krishi logo"
              className="nav-logo-img"
            />
            <span className="nav-brand-name">Samatva Krishi</span>
          </div>
          <div className="nav-divider" />
          <ul className="nav-list">
            <li><a href="#about" className={`nav-item ${activeNav === 'about' ? 'active' : ''}`} onClick={(e) => { triggerHaptic(); scrollTo('about')(e); }}>About</a></li>
            <li><a href="#analyse" className={`nav-item ${activeNav === 'analyse' ? 'active' : ''}`} onClick={(e) => { triggerHaptic(); scrollTo('analyse')(e); }}>Analyse</a></li>
            {result && <li><a href="#results" className={`nav-item ${activeNav === 'results' ? 'active' : ''}`} onClick={(e) => { triggerHaptic(); scrollTo('results')(e); }}>Results</a></li>}
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
            Upload a leaf photo and our system instantly identifies<br />
            the problem, helping you protect your harvest.
          </p>
          <div className="hero-actions" style={{ transform: `translateY(${scrollY * 0.22}px)` }}>
            <Magnetic>
              <button className="btn-hero solid" onClick={(e) => { triggerHaptic(); scrollTo('analyse')(e); }}>
                Analyse a Leaf →
              </button>
            </Magnetic>
            <Magnetic>
              <button className="btn-hero ghost" onClick={(e) => { triggerHaptic(); scrollTo('about')(e); }}>
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

      <section className="about-section" id="about" >
        <div className="about-inner">
          <div className="about-header">
            <h2 className="about-title reveal-blur">
              What is Samatva Krishi,<br />
              <span style={{ color: '#10b981' }}>and what can it do?</span>
            </h2>
            <p className="about-sub reveal-blur">
              Samatva Krishi AI is a smart plant disease detection tool. Upload a<br/>
              photo of a tomato leaf and the app instantly tells you whether the plant<br/>
              is healthy or identifies the problem — helping farmers and gardeners<br/>
              act faster and more accurately.
            </p>
          </div>

          <div className="tech-grid reveal-scale" data-stagger="">
            {[
              {
                icon: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
                    <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /><path d="M12 2a10 10 0 1 0 10 10" strokeDasharray="4 4"/>
                  </svg>
                ),
                label: 'Fast Results',
                desc: 'Receive a precise diagnosis within seconds of uploading — no expertise or lab required.',
              },
              {
                icon: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                ),
                label: 'Wide Coverage',
                desc: 'Detects 10 distinct tomato leaf conditions — from early blight to mosaic virus — in one analysis.',
              },
              {
                icon: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
                    <path d="M2 22L12 12M2 22H12M2 22V12"/><path d="M12 12c0-5.523 4.477-10 10-10v10c0 5.523-4.477 10-10 10s-10-4.477-10-10h10z"/>
                  </svg>
                ),
                label: 'Treatment Plan',
                desc: 'Get clear, step-by-step guidance on both organic and chemical treatment options for your crop.',
              },
              {
                icon: (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
                    <circle cx="12" cy="12" r="10" /><polyline points="9 12 11 14 15 10" />
                  </svg>
                ),
                label: 'Confidence Score',
                desc: 'Every diagnosis includes a reliability percentage so you know exactly how trustworthy the result is.',
              },
            ].map(({ icon, label, desc }) => (
              <TiltCard className="tech-card reveal-child" key={label} onClick={triggerHaptic}>
                <div className="tech-card-content">
                  <div className="tech-icon-v2">
                    {icon}
                  </div>
                  <div className="tech-label-v2">{label}</div>
                  <div className="tech-desc-v2">{desc}</div>
                </div>
              </TiltCard>
            ))}
          </div>

          <div className="how-to-section reveal-blur">
            <div className="how-to-inner">
              <div className="pipeline-header">
                <div className="pipeline-eyebrow-v2">
                  <div className="eyebrow-line" />
                  HOW TO USE SAMATVA KRISHI AI
                </div>
                <div className="dot-pattern" />
              </div>

              <div className="pipeline-content">
                <div className="pipeline-steps-v3">
                  {[
                    {
                      num: '01',
                      icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" /></svg>,
                      title: 'Photograph your leaf',
                      body: 'Take a clear, well-lit photo of a single tomato leaf — JPEG or PNG format.',
                    },
                    {
                      num: '02',
                      icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>,
                      title: 'Upload your image',
                      body: 'Drag the photo into the upload zone, or simply click to browse from your device.',
                    },
                    {
                      num: '03',
                      icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>,
                      title: 'Run the analysis',
                      body: 'Hit Analyse — our model identifies the condition from 10 possible diagnoses in seconds.',
                    },
                    {
                      num: '04',
                      icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /></svg>,
                      title: 'Review your treatment plan',
                      body: 'Receive a full report with the disease name, confidence score, and recommended action steps.',
                    },
                  ].map(({ num, icon, title, body }, idx, arr) => (
                    <div className="pipeline-step-v3 reveal-child" key={num} onClick={triggerHaptic} style={{ cursor: 'pointer' }}>
                      <div className="psv3-left">
                        <div className="psv3-icon-box">
                          {icon}
                        </div>
                        <div className="psv3-num-badge">{num}</div>
                        {idx < arr.length - 1 && <div className="psv3-connector" />}
                      </div>
                      <div className="psv3-body">
                        <div className="psv3-title">{title}</div>
                        <div className="psv3-desc">{body}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="pipeline-decoration">
                  <svg width="300" height="400" viewBox="0 0 100 120" fill="none" preserveAspectRatio="xMidYMid meet">
                    <path d="M50 100C50 100 30 70 30 40C30 10 50 0 50 0C50 0 70 10 70 40C70 70 50 100 50 100Z" fill="#10b981" fillOpacity="0.03"/>
                    <path d="M50 100C50 100 70 80 85 80C100 80 100 90 100 90C100 90 95 105 80 110C65 115 50 100 50 100Z" fill="#10b981" fillOpacity="0.05"/>
                    <path d="M50 80C50 80 30 65 15 65C0 65 0 75 0 75C0 75 5 85 20 90C35 95 50 80 50 80Z" fill="#10b981" fillOpacity="0.04"/>
                    <path d="M50 100L50 120" stroke="#10b981" strokeWidth="0.5" strokeOpacity="0.1"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>

          <div className="conditions-section reveal-blur">
            <div className="conditions-header">
              <div className="conditions-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2"><path d="M12 2L4.5 20.29C5.21 20.73 6.07 21 7 21C10.87 21 14 17.87 14 14V2H12ZM14 14C14 17.87 17.13 21 21 21C21.93 21 22.79 20.73 23.5 20.29L16 2V14Z"/></svg>
              </div>
              <div className="conditions-title">DETECTABLE CONDITIONS</div>
              <div className="conditions-badge">10 conditions</div>
            </div>
            <div className="conditions-grid">
              {/* Detectable conditions pills with haptics */}
              {[
                { name: 'Bacterial Spot', dot: '#ef4444' },
                { name: 'Early Blight', dot: '#10b981' },
                { name: 'Late Blight', dot: '#10b981' },
                { name: 'Leaf Mold', dot: '#047857' },
                { name: 'Septoria Leaf Spot', dot: '#34d399' },
                { name: 'Spider Mites', dot: '#10b981' },
                { name: 'Target Spot', dot: '#065f46' },
                { name: 'Tomato Mosaic Virus', dot: '#ef4444' },
                { name: 'Yellow Leaf Curl Virus', dot: '#14b8a6' },
                { name: 'Healthy', dot: '#84cc16' },
              ].map(({ name, dot }) => (
                <div className="condition-pill" key={name} onClick={triggerHaptic}>
                  <div className="condition-dot" style={{ background: dot }} />
                  {name}
                </div>
              ))}
            </div>
          </div>

        </div>{/* end about-inner */}
      </section >


      {/* ── STATS BAND REMOVED ── */}

            {/* ── UPLOAD SECTION with perspective tilt ── */}
      <section className="upload-section" id="analyse" >
        <div className="section-inner" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div className="section-eyebrow-lt reveal-blur" style={{ color: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
            — DISEASE DETECTION 
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 4h4v4H4zM16 4h4v4h-4zM4 16h4v4H4zM16 16h4v4h-4z"/><path d="M9 12h6M12 9v6"/></svg>
          </div>
          <h2 className="section-title reveal-blur" style={{ color: '#0b2414', fontSize: '3rem', fontWeight: '800', lineHeight: '1.1', marginBottom: '16px' }}>
            Drop your leaf.<br />Get your diagnosis.
          </h2>
          <p className="section-sub reveal-blur" style={{ color: '#4b5563', marginBottom: '40px', maxWidth: '500px', margin: '0 auto 40px', lineHeight: '1.6' }}>
            Our system has learned from thousands of tomato leaf photos<br/>to identify 10 different conditions. Simply upload and let it work.
          </p>

          <div className="upload-card reveal-tilt" style={{ background: '#fff', borderRadius: '20px', padding: '24px', boxShadow: '0 10px 40px rgba(0,0,0,0.08)', width: '100%', maxWidth: '700px' }}>
            <div
              className={`dropzone${dragging ? ' dragging' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              style={{ position: 'relative', border: '2px dashed #10b981', borderRadius: '12px', padding: '60px 20px', background: 'rgba(16,185,129,0.03)', cursor: 'pointer', overflow: 'hidden' }}
            >
              <input ref={inputRef} type="file" accept="image/jpeg,image/png"
                style={{ display: 'none' }} onChange={e => pick(e.target.files[0])} />
              
              <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className={`dz-ring ${preview ? 'success-pulse' : ''}`} style={{ width: '64px', height: '64px', background: '#e6f7ef', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
                  {preview
                    ? <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="32" height="32"><path d="M20 6L9 17l-5-5" /></svg>
                    : <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="32" height="32"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
                  }
                </div>
                {preview
                  ? <><div style={{ color: '#111827', fontWeight: '600' }} title={file?.name}>{file?.name}</div>
                    <div style={{ color: '#6b7280', fontSize: '0.9rem', marginTop: '8px' }}>Click to change image</div></>
                  : <><div style={{ color: '#111827', fontWeight: '500', fontSize: '1.1rem' }}>
                    <span style={{ color: '#10b981' }}>Click to upload</span> or drag &amp; drop
                  </div>
                    <div style={{ color: '#6b7280', fontSize: '0.9rem', marginTop: '8px' }}>JPEG or PNG · Tomato leaves only</div></>
                }
              </div>
              
              {/* Bottom wave decoration */}
              <div style={{ position: 'absolute', bottom: 0, left: 0, width: '100%', height: '40px', background: 'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 1200 120\' preserveAspectRatio=\'none\'%3E%3Cpath d=\'M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V120H0V95.8C59.71,118.08,130.83,115.11,192.17,100.2,236.43,89.44,279.37,71.55,321.39,56.44Z\' fill=\'%2310b981\' fill-opacity=\'0.15\'/%3E%3C/svg%3E") no-repeat center bottom', backgroundSize: '100% 100%' }} />
            </div>

            {preview && (
              <div className="preview" style={{ marginTop: '20px', borderRadius: '12px', overflow: 'hidden' }}>
                <img src={preview} alt="Leaf preview" className="preview-image-anim" style={{ width: '100%', maxHeight: '300px', objectFit: 'cover' }} />
              </div>
            )}

            <div className="upload-footer" style={{ marginTop: '24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 8px' }}>
              <div className="upload-meta" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#374151', fontSize: '0.95rem' }} title={file?.name}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                {file
                  ? <span><strong>{file.name}</strong> · {(file.size / 1024).toFixed(1)} KB</span>
                  : <span>No file selected</span>}
              </div>
              {file && (
                <button className="btn-analyse-inline" onClick={() => { triggerHaptic(); analyse(); }} disabled={loading} style={{ background: '#10b981', color: '#fff', border: 'none', padding: '8px 24px', borderRadius: '8px', fontWeight: '600', cursor: 'pointer' }}>
                  {loading
                    ? <><span className="spinner" style={{ width: '16px', height: '16px', marginRight: '8px', borderTopColor: '#fff', display: 'inline-block', verticalAlign: 'middle' }} />Analysing…</>
                    : 'Analyse'}
                </button>
              )}
            </div>

            {error && <div className="error-box" style={{ marginTop: '16px', padding: '12px', background: '#fef2f2', color: '#ef4444', borderRadius: '8px', fontSize: '0.9rem' }}>{error}</div>}
          </div>

          {/* Feature Badges below upload */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: '32px', marginTop: '32px', flexWrap: 'wrap' }}>
            {[
              { label: `AI Powered\nDetection`, icon: <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M12 8v8"/><path d="M8 12h8"/></svg> },
              { label: `Accurate\nResults`, icon: <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg> },
              { label: `Instant\nAnalysis`, icon: <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> },
              { label: `Secure &\nPrivate`, icon: <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg> }
            ].map((f, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', textAlign: 'left' }}>
                <div style={{ background: 'rgba(16,185,129,0.1)', padding: '10px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ width: '24px', height: '24px' }}>{f.icon}</div>
                </div>
                <div style={{ fontSize: '0.85rem', fontWeight: '600', color: '#111827', lineHeight: '1.3', whiteSpace: 'pre-line' }}>{f.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section >

      {/* ── RESULTS ── */}
      {
        result && (
          <section className="result-section" id="results">
            <div className="result-inner">
              <div className="result-header reveal-blur">
                <div className="result-tag"><div className="tag-dot" />Diagnosis</div>
                <div className="result-class-name">{formatClass(result.predicted_class)}</div>
              </div>

              <div className="result-cards" data-stagger="">
                <TiltCard className="rcard reveal-scale reveal-child" onClick={triggerHaptic}>
                  <div className="rcard-label">Predicted Disease</div>
                  <div className="rcard-val">{formatClass(result.predicted_class)}</div>
                </TiltCard>
                <TiltCard className="rcard reveal-scale reveal-child" style={{ transitionDelay: '100ms' }} onClick={triggerHaptic}>
                  <div className="rcard-label">Certainty Score</div>
                  <div className="rcard-val green">{result.confidence}</div>
                </TiltCard>
              </div>

              {/* LLM Recommendations Section */}
              <div className="llm-section reveal-blur">
                <div className="llm-header">
                  <div className="llm-badge-v2">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.6H22l-6.2 4.5L18.2 22l-6.2-4.5L5.8 22l2.4-7.9L2 9.6h7.6L12 2z"/></svg>
                    SMART ASSISTANT
                  </div>
                  <h3 className="llm-title-v2">
                    Treatment <span style={{ color: '#10b981' }}>Recommendations</span>
                  </h3>
                  <div className="llm-divider">
                    <div className="line" />
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="#10b981"><path d="M12 2L4.5 20.29C5.21 20.73 6.07 21 7 21C10.87 21 14 17.87 14 14V2H12ZM14 14C14 17.87 17.13 21 21 21C21.93 21 22.79 20.73 23.5 20.29L16 2V14Z"/></svg>
                    <div className="line" />
                  </div>
                </div>

                {recLoading ? (
                  <div className="llm-loading">
                    <div className="spinner" style={{ width: 24, height: 24, borderTopColor: 'var(--accent)' }} />
                    <span>Consulting plant expert...</span>
                  </div>
                ) : recommendations ? (
                  <>
                    <div className="llm-grid-v2" data-stagger="">
                      <div className="llm-card-v2 reveal-child" onClick={triggerHaptic} style={{ cursor: 'pointer' }}>
                        <div className="llm-card-header">
                          <div className="llm-icon-box symptoms">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                          </div>
                          <div className="llm-card-meta">
                            <div className="llm-tag symptoms">SYMPTOMS</div>
                            <div className="llm-line symptoms" />
                          </div>
                        </div>
                        <p className="llm-p">{recommendations.symptoms}</p>
                      </div>

                      <div className="llm-card-v2 reveal-child" onClick={triggerHaptic} style={{ cursor: 'pointer' }}>
                        <div className="llm-card-header">
                          <div className="llm-icon-box causes">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>
                          </div>
                          <div className="llm-card-meta">
                            <div className="llm-tag causes">CAUSES</div>
                            <div className="llm-line causes" />
                          </div>
                        </div>
                        <p className="llm-p">{recommendations.causes}</p>
                      </div>

                      <div className="llm-card-v2 reveal-child" onClick={triggerHaptic} style={{ cursor: 'pointer' }}>
                        <div className="llm-card-header">
                          <div className="llm-icon-box organic">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 8h-3c-1 0-3 2-3 2a7 7 0 0 1-3 8z" /></svg>
                          </div>
                          <div className="llm-card-meta">
                            <div className="llm-tag organic">ORGANIC TREATMENT</div>
                            <div className="llm-line organic" />
                          </div>
                        </div>
                        <p className="llm-p">{recommendations.organic}</p>
                      </div>

                      <div className="llm-card-v2 reveal-child" onClick={triggerHaptic} style={{ cursor: 'pointer' }}>
                        <div className="llm-card-header">
                          <div className="llm-icon-box chemical">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="6" y="8" width="12" height="12" rx="2"/><path d="M9 8V5a3 3 0 0 1 6 0v3"/></svg>
                          </div>
                          <div className="llm-card-meta">
                            <div className="llm-tag chemical">CHEMICAL TREATMENT</div>
                            <div className="llm-line chemical" />
                          </div>
                        </div>
                        <p className="llm-p">{recommendations.chemical}</p>
                      </div>
                    </div>

                    <div className="llm-footer-banner reveal-blur" onClick={triggerHaptic} style={{ cursor: 'pointer' }}>
                      <div className="llm-footer-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
                      </div>
                      <p>Follow these recommendations and monitor your plants regularly for the best results.</p>
                    </div>
                  </>
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
                src="/logo.jpeg"
                alt="Samatva Krishi logo"
                className="nav-logo-img"
              />
              Samatva Krishi AI
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
          <span>© 2026 Samatva Krishi AI. All rights reserved.</span>
          <span>Smart AI · Fast · Reliable · Secure</span>
        </div>
      </footer>
    </>
  )
}
