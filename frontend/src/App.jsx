import React, { useEffect, useState } from 'react'

const API_DEFAULT = import.meta.env.VITE_API_BASE_URL || ''

export default function App(){
  const [api, setApi] = useState(localStorage.getItem('api_base') || API_DEFAULT)
  const [tok, setTok] = useState(localStorage.getItem('api_tok') || '')
  const [out, setOut] = useState('')
  useEffect(()=>{ if(api) localStorage.setItem('api_base', api) },[api])
  useEffect(()=>{ localStorage.setItem('api_tok', tok) },[tok])
  const hdr = ()=> (tok ? { 'Authorization': 'Bearer ' + tok } : {})
  const base = api || (window.location.origin)
  const call = async (path, init={})=>{
    const res = await fetch(base + path, { ...init, headers: { 'Content-Type':'application/json', ...hdr() } })
    const text = await res.text()
    setOut(text)
  }
  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto', padding: 16 }}>
      <h3>简历筛选 · 前端</h3>
      <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
        <input placeholder='API_BASE_URL' value={api} onChange={e=>setApi(e.target.value)} style={{ padding:8 }} />
        <input placeholder='Bearer Token' value={tok} onChange={e=>setTok(e.target.value)} style={{ padding:8 }} />
        <button onClick={()=>call('/health')}>健康检查</button>
        <button onClick={()=>call('/config/industry_templates')}>模板查看</button>
        <button onClick={()=>call('/decision', { method:'POST', body: JSON.stringify({ job_desc:'示例岗位', top_k: 20, page:1, page_size:20 }) })}>生成推荐</button>
      </div>
      <hr />
      <MatchForm base={base} hdr={hdr} onFinish={setOut} />
      <FunnelForm base={base} hdr={hdr} onFinish={setOut} />
      <DecisionForm base={base} hdr={hdr} onFinish={setOut} />
      <hr />
      <h4>候选人上传（文本）</h4>
      <UploadText base={base} hdr={hdr} onFinish={setOut} />
      <h4>候选人上传（文件）</h4>
      <UploadFile base={base} hdr={hdr} onFinish={setOut} />
      <hr />
      <pre style={{ background:'#f7f7f7', padding:12, whiteSpace:'pre-wrap' }}>{out}</pre>
    </div>
  )
}

function UploadText({ base, hdr, onFinish }){
  const [text, setText] = useState('')
  const [name, setName] = useState('')
  const run = async ()=>{
    const fd = new FormData(); fd.append('text', text); if(name) fd.append('filename', name)
    const res = await fetch(base + '/uploads', { method:'POST', headers: hdr(), body: fd })
    onFinish(await res.text())
  }
  return (
    <div>
      <textarea placeholder='粘贴文本...' value={text} onChange={e=>setText(e.target.value)} style={{ width:'100%', height:120 }} />
      <div style={{ display:'flex', gap:8 }}>
        <input placeholder='文件名(可选)' value={name} onChange={e=>setName(e.target.value)} />
        <button onClick={run}>上传文本</button>
      </div>
    </div>
  )
}

function UploadFile({ base, hdr, onFinish }){
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const run = async ()=>{
    if(!file) return
    const fd = new FormData(); fd.append('file', file); if(name) fd.append('filename', name)
    const res = await fetch(base + '/uploads', { method:'POST', headers: hdr(), body: fd })
    onFinish(await res.text())
  }
  return (
    <div style={{ display:'flex', gap:8, alignItems:'center' }}>
      <input type='file' onChange={e=>setFile(e.target.files[0])} />
      <input placeholder='文件名(可选)' value={name} onChange={e=>setName(e.target.value)} />
      <button onClick={run}>上传文件</button>
    </div>
  )
}

function MatchForm({ base, hdr, onFinish }){
  const [r, setR] = useState('')
  const [j, setJ] = useState('')
  const run = async ()=>{
    const res = await fetch(base + '/match', { method:'POST', headers:{ 'Content-Type':'application/json', ...hdr() }, body: JSON.stringify({ resume_text: r, job_text: j }) })
    onFinish(await res.text())
  }
  return (
    <div>
      <h4>匹配</h4>
      <textarea placeholder='简历文本' value={r} onChange={e=>setR(e.target.value)} style={{ width:'100%', height:120 }} />
      <textarea placeholder='岗位文本' value={j} onChange={e=>setJ(e.target.value)} style={{ width:'100%', height:120 }} />
      <button onClick={run}>调用 /match</button>
    </div>
  )
}

function FunnelForm({ base, hdr, onFinish }){
  const [jd, setJd] = useState('')
  const [topk, setTopk] = useState(50)
  const [rules, setRules] = useState('[{"field":"years","operator":"gt","value":2}]')
  const run = async ()=>{
    let obj = []
    try{ obj = JSON.parse(rules) }catch{ obj = [] }
    const res = await fetch(base + '/filter', { method:'POST', headers:{ 'Content-Type':'application/json', ...hdr() }, body: JSON.stringify({ job_desc: jd, top_k: topk, custom_rules: obj }) })
    onFinish(await res.text())
  }
  return (
    <div>
      <h4>三级漏斗</h4>
      <textarea placeholder='岗位描述' value={jd} onChange={e=>setJd(e.target.value)} style={{ width:'100%', height:120 }} />
      <div style={{ display:'flex', gap:8 }}>
        <input type='number' value={topk} onChange={e=>setTopk(parseInt(e.target.value||'50',10))} />
        <input placeholder='规则JSON' value={rules} onChange={e=>setRules(e.target.value)} style={{ flex:1 }} />
        <button onClick={run}>调用 /filter</button>
      </div>
    </div>
  )
}

function DecisionForm({ base, hdr, onFinish }){
  const [jd, setJd] = useState('')
  const [topk, setTopk] = useState(50)
  const [page, setPage] = useState(1)
  const [ps, setPs] = useState(20)
  const [batch, setBatch] = useState('')
  const run = async ()=>{
    const lines = batch.split('\n').map(x=>x.trim()).filter(Boolean)
    if(lines.length>1){
      const res = await fetch(base + '/decision_batch', { method:'POST', headers:{ 'Content-Type':'application/json', ...hdr() }, body: JSON.stringify({ job_descs: lines, top_k: topk }) })
      onFinish(await res.text()); return
    }
    const res = await fetch(base + '/decision', { method:'POST', headers:{ 'Content-Type':'application/json', ...hdr() }, body: JSON.stringify({ job_desc: jd, top_k: topk, page, page_size: ps }) })
    onFinish(await res.text())
  }
  return (
    <div>
      <h4>决策与推荐</h4>
      <textarea placeholder='岗位描述（单个）' value={jd} onChange={e=>setJd(e.target.value)} style={{ width:'100%', height:120 }} />
      <textarea placeholder='批量岗位（每行一个）' value={batch} onChange={e=>setBatch(e.target.value)} style={{ width:'100%', height:120 }} />
      <div style={{ display:'flex', gap:8 }}>
        <input type='number' value={topk} onChange={e=>setTopk(parseInt(e.target.value||'50',10))} />
        <input type='number' value={page} onChange={e=>setPage(parseInt(e.target.value||'1',10))} />
        <input type='number' value={ps} onChange={e=>setPs(parseInt(e.target.value||'20',10))} />
        <button onClick={run}>调用 /decision 或 /decision_batch</button>
      </div>
    </div>
  )
}