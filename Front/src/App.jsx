import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSanitize from 'rehype-sanitize'
import { defaultSchema } from 'hast-util-sanitize'
import DOMPurify from 'dompurify'
import 'katex/dist/katex.min.css'

const katexAllowed = {
    ...defaultSchema,
    attributes: {
        ...defaultSchema.attributes,
        span: [...(defaultSchema.attributes?.span || []), 'class', 'className', 'style'],
        div: [...(defaultSchema.attributes?.div || []), 'class', 'className', 'style'],
        math: ['xmlns'],
        annotation: ['encoding']
    }
}

const WS_PORT = import.meta.env.VITE_WS_PORT || '8000'
const WS_URL = import.meta.env.VITE_WS_URL
    || `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:${WS_PORT}/ws/chat`

const THEMES = ['dark', 'light', 'pink', 'lightcoral', 'crimson', 'red', 'firebrick', 'darkred', 'yellow', 'orange', 'brown', 'gray', 'blue', 'green', 'purple']
const THEME_NAMES = {
    dark: '深色',
    light: '淺色',
    pink: '粉色',
    lightcoral: '淺珊瑚',
    crimson: '赤紅',
    red: '紅色',
    firebrick: '耐火磚',
    darkred: '深紅',
    yellow: '黃色',
    orange: '橘色',
    brown: '棕色',
    gray: '灰色',
    blue: '藍色',
    green: '綠色',
    purple: '紫色'
}

function validatePhone(value) {
    const digits = value.replace(/\D/g, '')
    return digits.length >= 8 && digits.length <= 15
}

function MarkdownViewer({ source, isInitial = false }) {
    const [displayText, setDisplayText] = useState('')

    useEffect(() => {
        if (!source) {
            setDisplayText('')
            return
        }

        // 只有初始訊息才做打字動畫（避免串流時 source 持續變長導致 interval 反覆重置與閃爍）
        if (isInitial) {
            let i = 0
            let isMounted = true
            const timer = setInterval(() => {
                if (isMounted && i < source.length) {
                    setDisplayText(source.substring(0, i + 1))
                    i++
                } else if (i >= source.length) {
                    clearInterval(timer)
                }
            }, 50) // 初始訊息用較慢速度，讓使用者看到打字效果

            return () => {
                isMounted = false
                clearInterval(timer)
            }
        }

        // 其他情境（包含 WS delta 串流）：直接顯示最新內容，避免「從頭打字」造成反覆消失/閃爍
        setDisplayText(source)
    }, [source, isInitial])

    const safeSource = typeof (displayText || source) === 'string'
        ? DOMPurify.sanitize(displayText || source)
        : (displayText || source)

    return (
        <ReactMarkdown
            children={safeSource}
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex, [rehypeSanitize, katexAllowed]]}
            components={{
                h1: ({ node, ...props }) => <h2 style={{ marginTop: '1em', marginBottom: '0.5em' }} {...props} />,
                h2: ({ node, ...props }) => <h3 style={{ marginTop: '0.8em', marginBottom: '0.4em' }} {...props} />,
                h3: ({ node, ...props }) => <h4 style={{ marginTop: '0.6em', marginBottom: '0.3em' }} {...props} />,
            }}
        />
    )
}

function LoadingIndicator() {
    return (
        <div className="msg-row assistant" style={{ opacity: 0.8 }}>
            <div className="msg assistant">
                <span className="loading-dots">AI 正在思考中</span>
            </div>
        </div>
    )
}

function ThemeSwitcher({ currentTheme, onThemeChange }) {
    return (
        <div className="theme-switcher">
            {THEMES.map(theme => (
                <button
                    key={theme}
                    className={`theme-btn ${theme} ${currentTheme === theme ? 'active' : ''}`}
                    onClick={() => onThemeChange(theme)}
                    title={`切換到${THEME_NAMES[theme]}主題`}
                    aria-label={`${THEME_NAMES[theme]}主題`}
                >
                    {theme === 'dark' && '🌙'}
                    {theme === 'light' && '☀️'}
                    {theme === 'pink' && '💖'}
                    {theme === 'lightcoral' && '🌸'}
                    {theme === 'crimson' && '🩸'}
                    {theme === 'red' && '❤️'}
                    {theme === 'firebrick' && '🧱'}
                    {theme === 'darkred' && '🩵'}  
                    {theme === 'yellow' && '⭐'}
                    {theme === 'orange' && '🔥'}
                    {theme === 'brown' && '🤎'}
                    {theme === 'gray' && '🩶'}
                    {theme === 'blue' && '💙'}
                    {theme === 'green' && '💚'}
                    {theme === 'purple' && '💜'}
                    <span className="theme-tooltip">{THEME_NAMES[theme]}</span>
                </button>
            ))}
        </div>
    )
}

export default function App() {
    // ===== 狀態管理 =====
    const [messages, setMessages] = useState([
        { id: 1, role: 'assistant', text: '歡迎！請輸入你的問題。' }
    ])
    const [input, setInput] = useState('')
    const [profileForm, setProfileForm] = useState({ name: '', phone: '' })
    const [profileError, setProfileError] = useState('')
    const [userProfile, setUserProfile] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [isConversationEnded, setIsConversationEnded] = useState(false)
    const [isAgentMode, setIsAgentMode] = useState(false)
    const [isComposing, setIsComposing] = useState(false)
    const [theme, setTheme] = useState(() => {
        return localStorage.getItem('chatTheme') || 'dark'
    })

    // ===== Refs =====
    const panelRef = useRef(null)
    const wsRef = useRef(null)
    const pendingAssistantId = useRef(null)
    const reconnectAttempts = useRef(0)
    const heartbeatRef = useRef({ timer: null, missed: 0 })
    const bufferRef = useRef('')
    const flushTimerRef = useRef(null)
    const NEXT_ID = () => Date.now() + Math.floor(Math.random() * 1000)

    // ===== 主題切換 =====
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme)
        localStorage.setItem('chatTheme', theme)
    }, [theme])

    function handleThemeChange(newTheme) {
        setTheme(newTheme)
    }

    // ===== 自動捲軸 =====
    useEffect(() => {
        if (panelRef.current) {
            setTimeout(() => {
                panelRef.current.scrollTop = panelRef.current.scrollHeight
            }, 0)
        }
    }, [messages])

    // ===== WebSocket 連線 =====
    const connectWs = useCallback((url = WS_URL) => {
        const existing = wsRef.current
        if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
            return
        }

        try {
            const ws = new WebSocket(url)
            wsRef.current = ws

            ws.onopen = () => {
                reconnectAttempts.current = 0
                startHeartbeat()
            }

            ws.onmessage = (evt) => {
                try {
                    const payload = JSON.parse(evt.data)
                    if (payload?.type === 'pong') {
                        heartbeatRef.current.missed = 0
                        return
                    }
                    handleWsPayload(payload)
                } catch (err) {
                    console.error('[ws] parse error', err)
                }
            }

            ws.onclose = (ev) => {
                stopHeartbeat()
                setIsLoading(false)
                pendingAssistantId.current = null
                clearFlushTimer()
                if (!ev.wasClean) scheduleReconnect(url)
            }

            ws.onerror = (e) => {
                console.error('[ws] error', e)
            }
        } catch (e) {
            console.error('[ws] connect failed', e)
            scheduleReconnect(url)
        }
    }, [])

    function waitForWsOpen(ws, timeout = 3000) {
        return new Promise((resolve, reject) => {
            if (!ws) return reject(new Error('No WebSocket'))
            if (ws.readyState === WebSocket.OPEN) return resolve()

            const onOpen = () => { cleanup(); resolve() }
            const onClose = () => { cleanup(); reject(new Error('WebSocket closed')) }
            const onError = (err) => { cleanup(); reject(err || new Error('WebSocket error')) }
            const timer = setTimeout(() => { cleanup(); reject(new Error('Timeout')) }, timeout)

            function cleanup() {
                clearTimeout(timer)
                ws.removeEventListener('open', onOpen)
                ws.removeEventListener('close', onClose)
                ws.removeEventListener('error', onError)
            }

            ws.addEventListener('open', onOpen)
            ws.addEventListener('close', onClose)
            ws.addEventListener('error', onError)
        })
    }

    function startHeartbeat() {
        stopHeartbeat()
        heartbeatRef.current.missed = 0
        heartbeatRef.current.timer = setInterval(() => {
            const ws = wsRef.current
            if (!ws || ws.readyState !== WebSocket.OPEN) return

            try {
                ws.send(JSON.stringify({ type: 'ping' }))
                heartbeatRef.current.missed += 1
                if (heartbeatRef.current.missed > 2) {
                    ws.close()
                }
            } catch (e) {
                console.error('[ws] heartbeat error', e)
            }
        }, 20000)
    }

    function stopHeartbeat() {
        if (heartbeatRef.current.timer) {
            clearInterval(heartbeatRef.current.timer)
            heartbeatRef.current.timer = null
        }
        heartbeatRef.current.missed = 0
    }

    function scheduleReconnect(url = WS_URL) {
        reconnectAttempts.current = Math.min(10, reconnectAttempts.current + 1)
        const delay = Math.min(30000, 200 * (2 ** reconnectAttempts.current))
        setTimeout(() => connectWs(url), delay)
    }

    // ===== 訊息 Buffer =====
    function flushBufferToMessage() {
        const text = bufferRef.current
        if (!text) return

        bufferRef.current = ''
        const aid = pendingAssistantId.current
        if (!aid) {
            const newId = NEXT_ID()
            pendingAssistantId.current = newId
            setMessages(prev => [...prev, { id: newId, role: 'assistant', text }])
            return
        }

        setMessages(prev => prev.map(m => m.id === aid ? { ...m, text: m.text + text } : m))
    }

    function ensureFlushTimer() {
        if (flushTimerRef.current) return
        flushTimerRef.current = setInterval(() => {
            flushBufferToMessage()
        }, 80)
    }

    function clearFlushTimer() {
        if (flushTimerRef.current) {
            clearInterval(flushTimerRef.current)
            flushTimerRef.current = null
        }
    }

    function extractDeltaText(payload) {
        if (!payload || typeof payload !== 'object') return ''
        const candidates = [payload.text, payload.response, payload.response_text, payload.output, payload.content]
        for (const v of candidates) {
            if (typeof v === 'string' && v.length > 0) return v
        }
        return ''
    }

    // ===== WebSocket 訊息處理 =====
    function handleWsPayload(payload) {
        if (!payload || typeof payload !== 'object') return

        if (payload.type === 'conversation_summary') {
            const summaryText = payload.summary || '對話摘要產生失敗。'
            setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: `${summaryText}` }])
            return
        }

        if (payload.type === 'conversation_ended') {
            setIsConversationEnded(true)
            setIsLoading(false)
            pendingAssistantId.current = null
            clearFlushTimer()
            return
        }

        if (payload.type === 'idle_warning') {
            setMessages(prev => [...prev, {
                id: NEXT_ID(),
                role: 'assistant',
                text: '⏰ 若 1 分鐘內沒有新對話，對話將自動關閉。'
            }])
            return
        }

        if (payload.type === 'agent_trace') {
            console.debug('[agent] trace', payload)
            return
        }

        if (payload.type === 'agent_final') {
            const finalText = payload.text || ''
            const aid = pendingAssistantId.current
            if (aid) {
                setMessages(prev => prev.map(m => m.id === aid ? { ...m, text: finalText } : m))
            } else {
                setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: finalText }])
            }
            pendingAssistantId.current = null
            setIsLoading(false)
            clearFlushTimer()
            return
        }

        if (payload.type === 'delta') {
            const delta = payload.text || ''
            bufferRef.current += delta
            ensureFlushTimer()
            return
        }

        const text = extractDeltaText(payload)
        if (text) {
            bufferRef.current += text
            ensureFlushTimer()
            if (payload.done === true) {
                flushBufferToMessage()
                pendingAssistantId.current = null
                setIsLoading(false)
                clearFlushTimer()
            }
            return
        }

        if (payload.type === 'done' || payload.done === true) {
            flushBufferToMessage()
            pendingAssistantId.current = null
            setIsLoading(false)
            clearFlushTimer()
            return
        }

        if (payload.type === 'error') {
            const errText = `❌ 錯誤：${payload.error || '伺服器錯誤'}`
            const aid = pendingAssistantId.current
            if (aid) {
                setMessages(prev => prev.map(m => m.id === aid ? { ...m, text: errText } : m))
            } else {
                setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: errText }])
            }
            pendingAssistantId.current = null
            setIsLoading(false)
            clearFlushTimer()
            return
        }

        console.warn('[ws] unknown payload', payload)
    }

    // ===== 發送訊息 =====
    async function sendMessage() {
        if (!userProfile) return
        if (isConversationEnded) return

        const trimmed = input.trim()
        if (!trimmed) return

        if (pendingAssistantId.current) {
            return
        }

        const userMsg = { id: NEXT_ID(), role: 'user', text: trimmed }
        setMessages(prev => [...prev, userMsg])
        setInput('')
        setIsLoading(true)

        const assistantId = NEXT_ID()
        pendingAssistantId.current = assistantId
        setMessages(prev => [...prev, { id: assistantId, role: 'assistant', text: '' }])

        try {
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
                connectWs(WS_URL)
                await waitForWsOpen(wsRef.current, 5000)
            }

            const ws = wsRef.current
            const payload = {
                model: 'CS_AgentV12',
                messages: [{ role: 'user', content: trimmed }],
                mode: isAgentMode ? 'agent' : 'chat',
                user_info: userProfile
            }
            ws.send(JSON.stringify(payload))
        } catch (err) {
            console.error('[send] error', err)
            setMessages(prev => prev.map(m => m.id === assistantId
                ? { ...m, text: '❌ 連線失敗，請檢查網路並稍後再試。' }
                : m
            ))
            pendingAssistantId.current = null
            setIsLoading(false)
            clearFlushTimer()
        }
    }

    // ===== 結束對話 =====
    function endConversation() {
        if (isConversationEnded) return
        setIsLoading(true)

        try {
            const ws = wsRef.current
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'end_conversation' }))
                return
            }

            setIsConversationEnded(true)
            setIsLoading(false)
            setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: '✅ 對話已結束。' }])
        } catch (e) {
            console.error('[end] error', e)
            setIsLoading(false)
        }
    }

    // ===== 生命週期 =====
    useEffect(() => {
        if (!userProfile) return
        connectWs(WS_URL)

        return () => {
            try {
                if (wsRef.current) wsRef.current.close()
            } catch (e) {
                console.debug('[cleanup] ws error', e)
            }
            stopHeartbeat()
            clearFlushTimer()
        }
    }, [connectWs, userProfile])

    // ===== Profile 提交 =====
    function submitProfile(e) {
        e.preventDefault()
        const name = profileForm.name.trim()
        const phone = profileForm.phone.trim()

        if (!name || !phone) {
            setProfileError('❌ 請輸入姓名與電話')
            return
        }

        if (!validatePhone(phone)) {
            setProfileError('❌ 請輸入有效電話號碼（8-15 位數字）')
            return
        }

        setProfileError('')
        setUserProfile({ name, phone })
        setIsConversationEnded(false)
        // 保留原始歡迎訊息格式，加上使用者名稱
        setMessages([{
            id: NEXT_ID(),
            role: 'assistant',
            text: `🎉 歡迎 ${name}！請輸入你的問題。`
        }])
    }

    // ===== 條件渲染：Profile 頁面 =====
    if (!userProfile) {
        return (
            <div className="app profile-page">
                <main className="profile-card">
                    <h1>請先留資料以獲得完整的服務</h1>
                    <form className="profile-form" onSubmit={submitProfile}>
                        <label className="profile-field">
                            <span>👤 姓名</span>
                            <input
                                type="text"
                                value={profileForm.name}
                                onChange={e => setProfileForm(prev => ({ ...prev, name: e.target.value }))}
                                placeholder="請輸入您的姓名"
                                maxLength={50}
                                required
                                aria-label="姓名輸入"
                            />
                        </label>
                        <label className="profile-field">
                            <span>📞 電話</span>
                            <input
                                type="tel"
                                value={profileForm.phone}
                                onChange={e => setProfileForm(prev => ({ ...prev, phone: e.target.value }))}
                                placeholder="請輸入電話（8-15 位數字）"
                                maxLength={20}
                                required
                                aria-label="電話輸入"
                            />
                        </label>
                        {profileError && <div className="profile-error">{profileError}</div>}
                        <button className="btn-send profile-submit" type="submit" aria-label="開始對話">
                            ▶ 進入聊天
                        </button>
                    </form>
                </main>
            </div>
        )
    }

    // ===== 主聊天頁面 =====
    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="container">
                    <div className="title-card">
                        <h1>具情緒反應識別與提示注入防護之大語言模型客服系統研製</h1>

                    </div>
                    <label
                        className="agent-toggle"
                        title="Agent 模式啟用高級推理；一般模式用於快速回應"
                        aria-label="Agent 模式切換"
                    >
                        <input
                            type="checkbox"
                            checked={isAgentMode}
                            disabled={isLoading || isConversationEnded}
                            onChange={e => setIsAgentMode(e.target.checked)}
                            aria-label="Agent 模式開關"
                        />
                        <span className="agent-toggle-slider" aria-hidden="true" />
                        <span className="agent-toggle-text">
                            {isAgentMode ? '🧠 Agent 模式' : '💬 標準模式'}
                        </span>
                    </label>
                </div>
            </header>

            {/* Chat Area */}
            <main className="chat-area" role="main" aria-label="聊天區域">
                <div className="inner">
                    <div
                        className="chat-panel"
                        ref={panelRef}
                        role="log"
                        aria-label="聊天訊息"
                        aria-live="polite"
                        aria-atomic="false"
                    >
                        {messages.length === 0 ? (
                            <div className="chat-panel empty">
                                <p style={{ color: 'var(--text-tertiary)', textAlign: 'center' }}>
                                    💬 還沒有訊息，開始對話吧！
                                </p>
                            </div>
                        ) : (
                            <>
                                {messages.map((msg, index) => (
                                    <div
                                        key={msg.id}
                                        className={`msg-row ${msg.role === 'user' ? 'user' : 'assistant'}`}
                                        role="article"
                                        aria-label={`${msg.role === 'user' ? '您的' : 'AI 的'}訊息`}
                                    >
                                        <div className={`msg ${msg.role === 'user' ? 'user' : 'assistant'}`}>
                                            {msg.role === 'assistant' ? (
                                                <MarkdownViewer
                                                    source={msg.text}
                                                    isInitial={index === 0 && messages.length === 1}
                                                />
                                            ) : (
                                                msg.text
                                            )}
                                        </div>
                                    </div>
                                ))}
                                {isLoading && <LoadingIndicator />}
                            </>
                        )}
                    </div>
                </div>
            </main>

            {/* Composer */}
            <footer className="composer" role="contentinfo">
                <form
                    className="row"
                    onSubmit={e => {
                        e.preventDefault()
                        if (!isLoading && !isConversationEnded && input.trim()) {
                            sendMessage()
                        }
                    }}
                    aria-label="訊息輸入表單"
                >
                    <textarea
                        className="input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder="輸入您的問題或需求..."
                        disabled={isLoading || isConversationEnded}
                        aria-label="訊息輸入框"
                        rows={3}
                        onCompositionStart={() => setIsComposing(true)}
                        onCompositionEnd={() => setIsComposing(false)}
                        onKeyDown={e => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                if (isComposing || e.nativeEvent?.isComposing || e.keyCode === 229) {
                                    return
                                }
                                e.preventDefault()
                                if (!isLoading && !isConversationEnded && input.trim()) {
                                    sendMessage()
                                }
                            }
                        }}
                    />

                    <button
                        className="btn-end"
                        type="button"
                        disabled={isConversationEnded || isLoading}
                        onClick={endConversation}
                        aria-label="結束對話"
                        title="點擊結束本次對話"
                    >
                        🏁 結束
                    </button>

                    <button
                        className="btn-send"
                        type="submit"
                        disabled={isLoading || isConversationEnded || !input.trim()}
                        aria-label="發送訊息"
                        title="發送訊息（Shift+Enter 換行）"
                    >
                        {isLoading ? '⏳ 傳送中' : '📤 發送'}
                    </button>
                </form>

                {/* 主題切換器 */}
                <div style={{ marginTop: 'var(--spacing-md)', display: 'flex', justifyContent: 'center' }}>
                    <ThemeSwitcher currentTheme={theme} onThemeChange={handleThemeChange} />
                </div>
            </footer>
        </div>
    )
}