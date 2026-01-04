'use client'

import { useState, useRef, useEffect, useCallback, FormEvent, ChangeEvent } from 'react'
import Image from 'next/image'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

interface FormData {
  patientName: string
  patientId: string
  dateOfSurgery: string
  age: string
  laterality: 'OD' | 'OS' | ''
  manifestCylinder: string
  manifestAxis: string
  barrettKMagnitude: string
  barrettKAxis: string
  deltaKIol700Magnitude: string
  deltaKIol700Axis: string
  deltaTkIol700Magnitude: string
  deltaTkIol700Axis: string
  postAstigIol700Magnitude: string
  postAstigIol700Axis: string
  pentacamDeltaKMagnitude: string
  pentacamDeltaKAxis: string
  axialLength: string
}

interface PredictionResult {
  arcuate_type: 'None' | 'Single' | 'Paired'
  arcuate_code: number
  lri_length: number | null
  lri_axis: number | null
  num_arcuates: number
  recommendation: string
}

type PredictionMode = 'auto' | 'single' | 'paired'

const initialFormData: FormData = {
  patientName: '', patientId: '', dateOfSurgery: '', age: '', laterality: '',
  manifestCylinder: '', manifestAxis: '', barrettKMagnitude: '', barrettKAxis: '',
  deltaKIol700Magnitude: '', deltaKIol700Axis: '', deltaTkIol700Magnitude: '', deltaTkIol700Axis: '',
  postAstigIol700Magnitude: '', postAstigIol700Axis: '', pentacamDeltaKMagnitude: '', pentacamDeltaKAxis: '',
  axialLength: '',
}

// Training data bounds - values outside these ranges were not seen during model training
const TRAINING_BOUNDS: Record<string, { min: number; max: number; label: string }> = {
  age: { min: 25, max: 87, label: 'Age' },
  manifestCylinder: { min: -3.25, max: 0, label: 'Manifest Cylinder' },
  barrettKMagnitude: { min: 0, max: 1.94, label: 'Barrett Integrated-K Magnitude' },
  deltaKIol700Magnitude: { min: 0, max: 2.67, label: 'ΔK IOL 700 Magnitude' },
  deltaTkIol700Magnitude: { min: 0, max: 2.63, label: 'ΔTK IOL 700 Magnitude' },
  postAstigIol700Magnitude: { min: 0, max: 0.99, label: 'Post. Astigmatism Magnitude' },
  pentacamDeltaKMagnitude: { min: 0, max: 2.30, label: 'Pentacam ΔK Magnitude' },
  axialLength: { min: 20.04, max: 29.43, label: 'Axial Length' },
}

export default function Home() {
  const [formData, setFormData] = useState<FormData>(initialFormData)
  const [predictionMode, setPredictionMode] = useState<PredictionMode>('auto')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [outOfBoundsWarnings, setOutOfBoundsWarnings] = useState<string[]>([])
  const [showComingSoon, setShowComingSoon] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Handle click on disabled mode buttons
  const handleDisabledModeClick = () => {
    setShowComingSoon(true)
    setTimeout(() => setShowComingSoon(false), 3000)
  }

  // Check for out-of-bounds values based on training data
  const checkBounds = useCallback(() => {
    const warnings: string[] = []
    for (const [field, bounds] of Object.entries(TRAINING_BOUNDS)) {
      const value = formData[field as keyof FormData]
      if (value !== '') {
        const numValue = parseFloat(value)
        if (!isNaN(numValue)) {
          if (numValue < bounds.min) {
            warnings.push(`${bounds.label}: ${numValue} is below training range (min: ${bounds.min})`)
          } else if (numValue > bounds.max) {
            warnings.push(`${bounds.label}: ${numValue} is above training range (max: ${bounds.max})`)
          }
        }
      }
    }
    setOutOfBoundsWarnings(warnings)
  }, [formData])

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
    if (result) setResult(null)
  }

  const drawArcuates = useCallback(() => {
    if (!canvasRef.current || !result || !formData.laterality) return
    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    const canvas = canvasRef.current
    const img = new window.Image()
    img.src = formData.laterality === 'OD' ? '/righteyetemplate.jpg' : '/lefteyetemplate.jpg'

    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      if (result.arcuate_type === 'None' || result.lri_length === null || result.lri_axis === null) return

      const radius = canvas.width * 0.35
      const center = { x: canvas.width / 2, y: canvas.height / 2 }
      const halfArcLength = (result.lri_length / 2) * (Math.PI / 180)
      const axisRadians = formData.laterality === 'OD' 
        ? -result.lri_axis * (Math.PI / 180)
        : -(180 - result.lri_axis) * (Math.PI / 180)

      ctx.beginPath()
      ctx.arc(center.x, center.y, radius, axisRadians - halfArcLength, axisRadians + halfArcLength)
      ctx.strokeStyle = '#FFFF00'
      ctx.lineWidth = canvas.width * 0.015
      ctx.stroke()

      if (result.arcuate_type === 'Paired') {
        ctx.beginPath()
        ctx.arc(center.x, center.y, radius, axisRadians + Math.PI - halfArcLength, axisRadians + Math.PI + halfArcLength)
        ctx.strokeStyle = '#FFA500'
        ctx.lineWidth = canvas.width * 0.015
        ctx.stroke()
      }
    }
  }, [result, formData.laterality])

  useEffect(() => { if (result) drawArcuates() }, [result, drawArcuates])
  useEffect(() => { checkBounds() }, [checkBounds])

  const validateForm = (): boolean => {
    const requiredFields: (keyof FormData)[] = [
      'age', 'laterality', 'manifestCylinder', 'manifestAxis', 'barrettKMagnitude', 'barrettKAxis',
      'deltaKIol700Magnitude', 'deltaKIol700Axis', 'deltaTkIol700Magnitude', 'deltaTkIol700Axis',
      'postAstigIol700Magnitude', 'postAstigIol700Axis', 'pentacamDeltaKMagnitude', 'pentacamDeltaKAxis', 'axialLength'
    ]
    for (const field of requiredFields) {
      if (!formData[field]) { setError('Please fill in all required fields'); return false }
    }
    const age = parseInt(formData.age)
    if (age < 21 || age > 120) { setError('Age must be between 21 and 120 years'); return false }
    const axialLength = parseFloat(formData.axialLength)
    if (axialLength < 20 || axialLength > 30) { setError('Axial length must be between 20 and 30 mm'); return false }
    return true
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setResult(null)
    if (!validateForm()) return

    setIsLoading(true)
    try {
      const apiData = {
        age: parseInt(formData.age), laterality: formData.laterality,
        manifest_cylinder: parseFloat(formData.manifestCylinder), manifest_axis: parseFloat(formData.manifestAxis),
        barrett_k_magnitude: parseFloat(formData.barrettKMagnitude), barrett_k_axis: parseFloat(formData.barrettKAxis),
        delta_k_iol700_magnitude: parseFloat(formData.deltaKIol700Magnitude), delta_k_iol700_axis: parseFloat(formData.deltaKIol700Axis),
        delta_tk_iol700_magnitude: parseFloat(formData.deltaTkIol700Magnitude), delta_tk_iol700_axis: parseFloat(formData.deltaTkIol700Axis),
        post_astig_iol700_magnitude: parseFloat(formData.postAstigIol700Magnitude), post_astig_iol700_axis: parseFloat(formData.postAstigIol700Axis),
        pentacam_delta_k_magnitude: parseFloat(formData.pentacamDeltaKMagnitude), pentacam_delta_k_axis: parseFloat(formData.pentacamDeltaKAxis),
        axial_length: parseFloat(formData.axialLength),
      }

      let endpoint = `${API_URL}/predict_warring`
      if (predictionMode === 'single') endpoint += '/single'
      else if (predictionMode === 'paired') endpoint += '/paired'

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(apiData),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Server error: ${response.status}`)
      }

      setResult(await response.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <header className="text-center mb-6">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Image src="/logo.png" alt="LRI Calculator" width={48} height={48} priority />
            <h1 className="text-3xl font-bold text-primary tracking-tight">LRI Calculator</h1>
          </div>
          <p className="text-slate-500">Laser LRI Prediction Tool</p>
        </header>

        {/* Disclaimer */}
        <div className="flex items-center gap-3 px-5 py-3.5 mb-6 bg-gradient-to-r from-blue-50 to-sky-50 border border-blue-200 rounded-xl text-primary text-sm shadow-sm no-print">
          <span className="text-lg">ℹ️</span>
          This web application is intended for investigational purposes only.
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="px-6 py-4 bg-gradient-to-r from-primary to-primary-light text-white no-print">
            <h2 className="text-lg font-semibold">Enter Case Details</h2>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-6 no-print">
            {/* Patient Information */}
            <fieldset className="border border-slate-200 rounded-xl p-5 bg-slate-50/50">
              <legend className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2">Patient Information</legend>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Patient Name</label>
                  <input type="text" name="patientName" value={formData.patientName} onChange={handleChange} className="input-field" placeholder="Enter patient name" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Patient ID</label>
                  <input type="text" name="patientId" value={formData.patientId} onChange={handleChange} className="input-field" placeholder="Enter patient ID" />
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Date of Surgery</label>
                  <input type="date" name="dateOfSurgery" value={formData.dateOfSurgery} onChange={handleChange} className="input-field" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Age (years) *</label>
                  <input type="number" name="age" min="21" max="120" value={formData.age} onChange={handleChange} className="input-field" placeholder="21-120" required />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Eye *</label>
                  <div className="flex gap-4 pt-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="radio" name="laterality" value="OD" checked={formData.laterality === 'OD'} onChange={handleChange} className="w-4 h-4 accent-primary" required />
                      <span className="text-sm text-slate-700">Right (OD)</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="radio" name="laterality" value="OS" checked={formData.laterality === 'OS'} onChange={handleChange} className="w-4 h-4 accent-primary" />
                      <span className="text-sm text-slate-700">Left (OS)</span>
                    </label>
                  </div>
                </div>
              </div>
            </fieldset>

            {/* Manifest Refraction */}
            <fieldset className="border border-slate-200 rounded-xl p-5 bg-slate-50/50">
              <legend className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2">Manifest Refraction</legend>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Cylinder (D) *</label>
                  <input type="number" name="manifestCylinder" step="0.25" max="0" value={formData.manifestCylinder} onChange={handleChange} className="input-field" placeholder="e.g., -1.50" required />
                  <span className="text-xs text-slate-400">Negative notation</span>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Axis (°) *</label>
                  <input type="number" name="manifestAxis" min="1" max="180" value={formData.manifestAxis} onChange={handleChange} className="input-field" placeholder="1-180" required />
                </div>
              </div>
            </fieldset>

            {/* Barrett Integrated-K */}
            <fieldset className="border border-slate-200 rounded-xl p-5 bg-slate-50/50">
              <legend className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2">Barrett Integrated-K</legend>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Magnitude (D) *</label>
                  <input type="number" name="barrettKMagnitude" step="0.01" min="0" value={formData.barrettKMagnitude} onChange={handleChange} className="input-field" placeholder="e.g., 1.25" required />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">Axis (°) *</label>
                  <input type="number" name="barrettKAxis" min="1" max="180" value={formData.barrettKAxis} onChange={handleChange} className="input-field" placeholder="1-180" required />
                </div>
              </div>
            </fieldset>

            {/* IOL 700 Measurements - Now includes Axial Length */}
            <fieldset className="border border-slate-200 rounded-xl p-5 bg-slate-50/50">
              <legend className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2">IOL 700 Measurements</legend>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">ΔK Magnitude (D) *</label>
                    <input type="number" name="deltaKIol700Magnitude" step="0.01" value={formData.deltaKIol700Magnitude} onChange={handleChange} className="input-field" placeholder="e.g., 0.45" required />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">ΔK Axis (°) *</label>
                    <input type="number" name="deltaKIol700Axis" min="1" max="180" value={formData.deltaKIol700Axis} onChange={handleChange} className="input-field" placeholder="1-180" required />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">ΔTK Magnitude (D) *</label>
                    <input type="number" name="deltaTkIol700Magnitude" step="0.01" value={formData.deltaTkIol700Magnitude} onChange={handleChange} className="input-field" placeholder="e.g., 0.52" required />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">ΔTK Axis (°) *</label>
                    <input type="number" name="deltaTkIol700Axis" min="1" max="180" value={formData.deltaTkIol700Axis} onChange={handleChange} className="input-field" placeholder="1-180" required />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">Post. Astig. Magnitude (D) *</label>
                    <input type="number" name="postAstigIol700Magnitude" step="0.01" value={formData.postAstigIol700Magnitude} onChange={handleChange} className="input-field" placeholder="e.g., 0.38" required />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">Post. Astig. Axis (°) *</label>
                    <input type="number" name="postAstigIol700Axis" min="1" max="180" value={formData.postAstigIol700Axis} onChange={handleChange} className="input-field" placeholder="1-180" required />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-600">Axial Length (mm) *</label>
                    <input type="number" name="axialLength" step="0.01" min="20" max="30" value={formData.axialLength} onChange={handleChange} className="input-field" placeholder="20-30" required />
                  </div>
                </div>
              </div>
            </fieldset>

            {/* Pentacam */}
            <fieldset className="border border-slate-200 rounded-xl p-5 bg-slate-50/50">
              <legend className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2">Pentacam</legend>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">ΔK Magnitude (D) *</label>
                  <input type="number" name="pentacamDeltaKMagnitude" step="0.01" value={formData.pentacamDeltaKMagnitude} onChange={handleChange} className="input-field" placeholder="e.g., 0.41" required />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-600">ΔK Axis (°) *</label>
                  <input type="number" name="pentacamDeltaKAxis" min="1" max="180" value={formData.pentacamDeltaKAxis} onChange={handleChange} className="input-field" placeholder="1-180" required />
                </div>
              </div>
            </fieldset>

            {/* Prediction Mode */}
            <fieldset className="border border-slate-200 rounded-xl p-5 bg-slate-50/50">
              <legend className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-2">Prediction Mode</legend>
              <div className="flex gap-3 mb-3">
                <button type="button" onClick={() => setPredictionMode('auto')} className={`mode-btn ${predictionMode === 'auto' ? 'mode-btn-active' : ''}`}>Auto Select</button>
                <button type="button" onClick={handleDisabledModeClick} className="mode-btn mode-btn-disabled cursor-not-allowed">Single</button>
                <button type="button" onClick={handleDisabledModeClick} className="mode-btn mode-btn-disabled cursor-not-allowed">Paired</button>
              </div>
              <p className="text-sm text-slate-500 italic">
                The model will automatically determine the optimal arcuate type.
              </p>
              {showComingSoon && (
                <p className="text-sm text-amber-600 mt-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                  ⏳ Single and Paired prediction modes will be added in a future update.
                </p>
              )}
            </fieldset>

            {/* Out of Bounds Warnings */}
            {outOfBoundsWarnings.length > 0 && (
              <div className="px-4 py-3 bg-amber-50 border border-amber-300 rounded-lg">
                <div className="flex items-center gap-2 mb-2 text-amber-700 font-medium">
                  <span className="text-lg">⚠️</span>
                  <span>Caution: Some inputs are outside training data range</span>
                </div>
                <ul className="text-sm text-amber-600 ml-7 space-y-1">
                  {outOfBoundsWarnings.map((warning, idx) => (
                    <li key={idx}>• {warning}</li>
                  ))}
                </ul>
                <p className="text-xs text-amber-500 mt-2 ml-7 italic">
                  Predictions may be less reliable for values not seen during AI model training.
                </p>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="flex items-center gap-3 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                <span className="text-lg">⚠️</span>{error}
              </div>
            )}

            {/* Submit */}
            <div className="flex justify-center pt-2">
              <button type="submit" className="btn-primary flex items-center gap-2" disabled={isLoading}>
                {isLoading ? (<><span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Calculating...</>) : 'Calculate Recommendation'}
              </button>
            </div>
          </form>

          {/* Print-only patient info */}
          <div className="print-only p-6">
            <h2 className="text-lg font-bold text-slate-800 mb-2">Patient Information</h2>
            <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
              <div><span className="text-slate-500">Name:</span> <strong>{formData.patientName || '-'}</strong></div>
              <div><span className="text-slate-500">ID:</span> <strong>{formData.patientId || '-'}</strong></div>
              <div><span className="text-slate-500">Eye:</span> <strong>{formData.laterality === 'OD' ? 'Right (OD)' : formData.laterality === 'OS' ? 'Left (OS)' : '-'}</strong></div>
              <div><span className="text-slate-500">Surgery Date:</span> <strong>{formData.dateOfSurgery || '-'}</strong></div>
              <div><span className="text-slate-500">Age:</span> <strong>{formData.age ? `${formData.age} yrs` : '-'}</strong></div>
              <div><span className="text-slate-500">Axial Length:</span> <strong>{formData.axialLength ? `${formData.axialLength} mm` : '-'}</strong></div>
            </div>
          </div>

          {/* Results */}
          {result && (
            <div className="p-6 bg-slate-100 border-t border-slate-200">
              {/* Out of bounds warning for print */}
              {outOfBoundsWarnings.length > 0 && (
                <div className="px-4 py-3 mb-4 bg-amber-50 border border-amber-300 rounded-lg print:bg-amber-50">
                  <div className="flex items-center gap-2 text-amber-700 font-medium text-sm">
                    <span>⚠️ Caution:</span>
                    <span>Some inputs outside training range - predictions may be less reliable</span>
                  </div>
                  <ul className="text-xs text-amber-600 ml-6 mt-1">
                    {outOfBoundsWarnings.map((warning, idx) => (
                      <li key={idx}>• {warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-5 mb-5 text-white">
                <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400 mb-2">Recommendation</h3>
                <p className="text-lg font-semibold text-yellow-400 mb-4">{result.recommendation}</p>
                {result.arcuate_type !== 'None' && (
                  <div className="grid grid-cols-4 gap-4 pt-4 border-t border-white/10">
                    <div><span className="text-xs uppercase tracking-wide text-slate-400">Type</span><p className="text-lg font-semibold font-mono">{result.arcuate_type}</p></div>
                    <div><span className="text-xs uppercase tracking-wide text-slate-400">Length</span><p className="text-lg font-semibold font-mono">{result.lri_length}°</p></div>
                    <div><span className="text-xs uppercase tracking-wide text-slate-400">Axis</span><p className="text-lg font-semibold font-mono">{result.lri_axis}°</p></div>
                    <div><span className="text-xs uppercase tracking-wide text-slate-400">Arcuates</span><p className="text-lg font-semibold font-mono">{result.num_arcuates}</p></div>
                  </div>
                )}
              </div>

              <div className="bg-white rounded-xl p-4 shadow-md">
                <div className="max-w-md mx-auto">
                  <canvas ref={canvasRef} width={1024} height={1024} className="w-full h-auto rounded-xl aspect-square" />
                </div>
                <div className="text-center py-3 mt-4 bg-slate-100 rounded-lg">
                  <span className="text-xl font-bold text-primary tracking-wider">{formData.laterality === 'OD' ? 'RIGHT EYE (OD)' : 'LEFT EYE (OS)'}</span>
                </div>
                {result.arcuate_type !== 'None' && (
                  <div className="flex justify-center gap-6 mt-4">
                    <div className="flex items-center gap-2"><span className="w-4 h-4 rounded bg-yellow-400 border border-black/10" /><span className="text-sm text-slate-600">Primary Arcuate</span></div>
                    {result.arcuate_type === 'Paired' && (<div className="flex items-center gap-2"><span className="w-4 h-4 rounded bg-orange-500 border border-black/10" /><span className="text-sm text-slate-600">Secondary Arcuate</span></div>)}
                  </div>
                )}
              </div>

              <div className="flex justify-center mt-5 no-print">
                <button onClick={() => window.print()} className="btn-secondary flex items-center gap-2">🖨️ Print Results</button>
              </div>
            </div>
          )}
        </div>

        <footer className="text-center py-8 text-slate-400 text-sm no-print">LRI Calculator © {new Date().getFullYear()}</footer>
      </div>
    </main>
  )
}
