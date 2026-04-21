import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSanitize from 'rehype-sanitize'
import { defaultSchema } from 'hast-util-sanitize'
import DOMPurify from 'dompurify'
import 'katex/dist/katex.min.css' // KaTeX 樣式

/**
 * App.jsx（前端聊天室核心）
 *
 * 設計重點：
 * 1) 使用 WebSocket 串流接收後端回覆，避免一次性等待整段內容。
 * 2) 透過 buffer + 定時 flush，降低 React 高頻 setState 造成的重繪成本。
 * 3) 以雙重防護做 Markdown/HTML 安全處理：
 *    - DOMPurify：先清理原始輸入內容
 *    - rehype-sanitize：再限制 Markdown 轉換後的 HTML 節點與屬性
 * 4) 支援 IME（中文輸入法）組字時 Enter 不誤送出的互動細節。
 */

// 擴充 sanitize schema，允許 KaTeX 會用到的 class 與 style
const katexAllowed = {
    ...defaultSchema,
    attributes: {
        ...defaultSchema.attributes,
        // 允許 span/div 的 class 與 style（KaTeX 會產生許多 span）
        span: [
            ...(defaultSchema.attributes && defaultSchema.attributes.span ? defaultSchema.attributes.span : []),
            'class',
            'className',
            'style'
        ],
        div: [
            ...(defaultSchema.attributes && defaultSchema.attributes.div ? defaultSchema.attributes.div : []),
            'class',
            'className',
            'style'
        ],
        // 若 KaTeX 生成 MathML，允許少量必要屬性
        math: ['xmlns'],
        annotation: ['encoding']
    }
}

const WS_URL = (window.location.protocol === 'https:' ? 'wss' : 'ws') + '://100.111.80.10:8000/ws/chat'

function validatePhone(value) {
    const digits = value.replace(/\D/g, '')
    return digits.length >= 8 && digits.length <= 15
}

/**
 * MarkdownViewer
 * - 支援 GFM 與 LaTeX（remark-math + rehype-katex）
 * - 先用 DOMPurify 清理輸入 markdown（避免惡意 HTML），再由 rehypeSanitize（katexAllowed）保護 KaTeX 生成的 HTML
 */
function MarkdownViewer({ source }) {
    // source 很可能來自後端流式累積；如果來源是未信任的，先用 DOMPurify sanitize
    const safeSource = typeof source === 'string' ? DOMPurify.sanitize(source) : source

    return (
        <ReactMarkdown
            children={safeSource}
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[
                rehypeKatex,                 // 先把 LaTeX 轉成 HTML
                [rehypeSanitize, katexAllowed] // 然後以擴充過的 schema 做 sanitize（允許 KaTeX 標籤/屬性）
            ]}
        />
    )
}

export default function App() {
    // 聊天訊息列表：user / assistant 混合顯示
    const [messages, setMessages] = useState([
        { id: 1, role: 'assistant', text: '歡迎！請輸入你的問題。' }
    ])
    // 使用者輸入框內容
    const [input, setInput] = useState('')
    const [profileForm, setProfileForm] = useState({ name: '', phone: '' })
    const [profileError, setProfileError] = useState('')
    const [userProfile, setUserProfile] = useState(null)
    // 是否正在等待/接收後端回覆（控制送出按鈕與 UI 狀態）
    const [isLoading, setIsLoading] = useState(false)
    const [isConversationEnded, setIsConversationEnded] = useState(false)
    const [isComposing, setIsComposing] = useState(false) // ✅ 新增：IME 組字狀態
    // 聊天面板 DOM 參照，用於自動滾動到底部
    const panelRef = useRef(null)

    // WebSocket 實體參照
    const wsRef = useRef(null)
    // 當前「尚未完成」的 assistant 訊息 id（串流回填目標）
    const pendingAssistantId = useRef(null)
    // 斷線重連計數，用於 exponential backoff
    const reconnectAttempts = useRef(0)
    // 心跳控制：timer + 未收到 pong 的次數
    const heartbeatRef = useRef({ timer: null, missed: 0 })
    const bufferRef = useRef('')           // accumulate small deltas
    const flushTimerRef = useRef(null)
    // 用時間戳 + 隨機數生成前端訊息 id（避免碰撞）
    const NEXT_ID = () => Date.now() + Math.floor(Math.random() * 1000)

    // auto-scroll
    useEffect(() => {
        if (panelRef.current) panelRef.current.scrollTop = panelRef.current.scrollHeight
    }, [messages])

    // 根據頁面協議動態選擇 ws / wss，避免 HTTPS 頁面混用不安全 ws
    const connectWs = useCallback((url = WS_URL) => {
        // 如果已連線或正在連線，就不重複建立，避免多條 socket 造成狀態錯亂
        const existing = wsRef.current
        if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) return

        try {
            const ws = new WebSocket(url)
            wsRef.current = ws

            ws.onopen = () => {
                // 連線成功後重置重連次數並啟動心跳機制
                reconnectAttempts.current = 0
                startHeartbeat()
            }

            ws.onmessage = (evt) => {
                try {
                    const payload = JSON.parse(evt.data)
                    if (payload && payload.type === 'pong') {
                        // 收到心跳回覆，將 missed 歸零
                        heartbeatRef.current.missed = 0
                        return
                    }
                    // 其餘訊息交由統一 handler（delta/done/error 等）處理
                    handleWsPayload(payload)
                } catch (err) {
                    console.error('[ws] parse error', err, evt.data)
                }
            }

            ws.onclose = (ev) => {
                stopHeartbeat()
                if (!ev.wasClean) scheduleReconnect(url)
            }

            ws.onerror = (e) => {
                console.error('[ws] error', e)
            }
        } catch (e) {
            console.error('[ws] connect failed', e)
            scheduleReconnect(url)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    function waitForWsOpen(ws, timeout = 3000) {
        // 把 WebSocket 事件轉為 Promise，讓 sendMessage 可 await 連線完成
        return new Promise((resolve, reject) => {
            if (!ws) return reject(new Error('No WebSocket'))
            if (ws.readyState === WebSocket.OPEN) return resolve()
            const onOpen = () => { cleanup(); resolve() }
            const onClose = () => { cleanup(); reject(new Error('WebSocket closed before open')) }
            const onError = (err) => { cleanup(); reject(err || new Error('WebSocket error before open')) }
            const timer = setTimeout(() => { cleanup(); reject(new Error('WebSocket open timeout')) }, timeout)
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
        // 先清理舊 timer，再重建，避免重複 setInterval
        stopHeartbeat()
        heartbeatRef.current.missed = 0
        heartbeatRef.current.timer = setInterval(() => {
            const ws = wsRef.current
            if (!ws || ws.readyState !== WebSocket.OPEN) return
            try {
                ws.send(JSON.stringify({ type: 'ping' }))
                heartbeatRef.current.missed += 1
                // 連續多次沒有 pong 視為死連線，主動 close 觸發重連流程
                if (heartbeatRef.current.missed > 2) {
                    ws.close()
                }
            } catch (e) {
                console.error('[ws] heartbeat send failed', e)
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
        const attempt = reconnectAttempts.current
        // 指數退避，避免重連風暴；最大延遲 30 秒
        const delay = Math.min(30000, 200 * 2 ** attempt)
        setTimeout(() => connectWs(url), delay)
    }

    // Flush bufferRef into the current pending assistant message
    function flushBufferToMessage() {
        const text = bufferRef.current
        if (!text) return
        bufferRef.current = ''
        const aid = pendingAssistantId.current
        if (!aid) {
            // 若不存在 pending assistant，建立新的 assistant 泡泡承接文字
            const newId = NEXT_ID()
            pendingAssistantId.current = newId
            setMessages(prev => [...prev, { id: newId, role: 'assistant', text }])
            return
        }
        // 仍在同一則 assistant 回覆中時，採 append 方式增量更新文字
        setMessages(prev => prev.map(m => m.id === aid ? { ...m, text: m.text + text } : m))
    }

    // start a periodic flush when streaming
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

    // unify incoming payload to a text delta
    function extractDeltaText(payload) {
        if (!payload || typeof payload !== 'object') return ''
        // 兼容多種後端欄位命名，提升前後端版本差異時的韌性
        const candidates = [
            payload.text,
            payload.response,
            payload.response_text,
            payload.output,
            payload.content
        ]
        for (const v of candidates) {
            if (typeof v === 'string' && v.length > 0) return v
        }
        if (typeof payload.thinking === 'string' && payload.thinking.trim() !== '') {
            // 思考內容若非最終輸出，不顯示在主訊息泡泡
            return ''
        }
        return ''
    }

    function handleWsPayload(payload) {
        if (!payload || typeof payload !== 'object') return
        if (payload.type === 'conversation_summary') {
            const summaryText = payload.summary || '本次對話摘要產生失敗。'
            setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: `### 對話摘要\n${summaryText}` }])
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
            const defaultRemainingSeconds = 60
            const parsedRemainingSeconds = Number(payload.remaining_seconds)
            const remainingSeconds = Number.isFinite(parsedRemainingSeconds) && parsedRemainingSeconds > 0
                ? parsedRemainingSeconds
                : defaultRemainingSeconds
            const remainingTimeText = remainingSeconds < 60
                ? `${Math.ceil(remainingSeconds)} 秒`
                : `${Math.ceil(remainingSeconds / 60)} 分鐘`
            setMessages(prev => [
                ...prev,
                { id: NEXT_ID(), role: 'assistant', text: `提醒：若 ${remainingTimeText}內沒有新對話，對話將自動關閉。` }
            ])
            return
        }

        if (payload.type === 'delta') {
            // 串流片段先進 buffer，再由 flush timer 批次更新 UI
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
            const errText = payload.error || '伺服器錯誤'
            const aid = pendingAssistantId.current
            if (aid) {
                // 如果有正在生成的 assistant 訊息，直接覆寫成錯誤訊息
                setMessages(prev => prev.map(m => m.id === aid ? { ...m, text: errText } : m))
            } else {
                // 否則新增一則 assistant 錯誤訊息，讓使用者看到失敗原因
                setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: errText }])
            }
            pendingAssistantId.current = null
            setIsLoading(false)
            clearFlushTimer()
            return
        }

        console.warn('[ws] unknown payload', payload)
    }

    async function sendMessage() {
        if (!userProfile) return
        if (isConversationEnded) return
        const trimmed = input.trim()
        if (!trimmed) return
        if (pendingAssistantId.current) {
            // 防止併發送出導致回覆交錯：上一則未完成時拒絕新請求
            setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: '請等待前一則回覆完畢' }])
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
                // 若 socket 尚未就緒，先建連線並等待 open
                connectWs(WS_URL)
                await waitForWsOpen(wsRef.current, 5000)
            }
        } catch (e) {
            // Connection error already handled by UI state
            console.error('WebSocket connection error:', e);
            setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, text: '無法連線，請稍後再試。' } : m))
            pendingAssistantId.current = null
            setIsLoading(false)
            return
        }

        try {
            const ws = wsRef.current
            const payload = {
                model: 'gpt-oss:20b',
                messages: [{ role: 'user', content: trimmed }],
                user_info: userProfile
            }
            // 請求格式與後端約定一致：model + messages[]
            ws.send(JSON.stringify(payload))
        } catch (err) {
            // Send error already handled by UI state
            console.error('WebSocket send error:', err);
            setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, text: '送出失敗，請稍後再試。' } : m))
            pendingAssistantId.current = null
            setIsLoading(false)
            clearFlushTimer()
        }
    }

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
            setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: '對話已結束。' }])
        } catch (e) {
            console.error('End conversation error:', e)
            setIsLoading(false)
            setMessages(prev => [...prev, { id: NEXT_ID(), role: 'assistant', text: '結束對話失敗，請稍後再試。' }])
        }
    }

    // connect on mount; cleanup on unmount
    useEffect(() => {
        if (!userProfile) return
        connectWs(WS_URL)
        return () => {
            try {
                if (wsRef.current) wsRef.current.close()
            } catch(e) {
                // WebSocket cleanup error - can be safely ignored during unmount
                console.debug('WebSocket cleanup error:', e);
            }
            // 確保離開頁面時停止背景 timer，避免記憶體洩漏
            stopHeartbeat()
            clearFlushTimer()
        }
    }, [connectWs, userProfile])

    function submitProfile(e) {
        e.preventDefault()
        const name = profileForm.name.trim()
        const phone = profileForm.phone.trim()
        if (!name || !phone) {
            setProfileError('請輸入姓名與電話')
            return
        }
        if (!validatePhone(phone)) {
            setProfileError('請輸入有效電話號碼')
            return
        }
        setProfileError('')
        setUserProfile({ name, phone })
        setIsConversationEnded(false)
        setMessages([{ id: NEXT_ID(), role: 'assistant', text: `歡迎 ${name}！請輸入你的問題。` }])
    }

    if (!userProfile) {
        return (
            <div className="app profile-page">
                <main className="profile-card">
                    <h1>進入聊天前請先留資料</h1>
                    <form className="profile-form" onSubmit={submitProfile}>
                        <label className="profile-field">
                            姓名
                            <input
                                type="text"
                                value={profileForm.name}
                                onChange={e => setProfileForm(prev => ({ ...prev, name: e.target.value }))}
                                placeholder="請輸入姓名"
                                maxLength={50}
                            />
                        </label>
                        <label className="profile-field">
                            電話
                            <input
                                type="tel"
                                value={profileForm.phone}
                                onChange={e => setProfileForm(prev => ({ ...prev, phone: e.target.value }))}
                                placeholder="請輸入電話"
                                maxLength={20}
                            />
                        </label>
                        {profileError && <div className="profile-error">{profileError}</div>}
                        <button className="btn-send profile-submit" type="submit">進入聊天</button>
                    </form>
                </main>
            </div>
        )
    }

    return (
        <div className="app">
            <header className="header">
                <div className="container">
                    <div>
                        <h1>智慧聊天機器人</h1>
                        <div className="meta">建立 Ollama 的智慧聊天系統</div>
                    </div>
                </div>
            </header>

            <main className="chat-area">
                <div className="inner">
                    <div className="chat-panel" ref={panelRef} style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                        {messages.length === 0 ? (
                            <div className="empty">還沒有訊息，請輸入開始對話。</div>
                        ) : (
                            messages.map(msg => (
                                <div
                                    key={msg.id}
                                    className={`msg-row ${msg.role === 'user' ? 'user' : 'assistant'}`}
                                >
                                    <div className={`msg ${msg.role === 'user' ? 'user' : 'assistant'}`}>
                                        {msg.role === 'assistant' ? (
                                            <MarkdownViewer source={msg.text} />
                                        ) : (
                                            msg.text
                                        )}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </main>

            <footer className="composer">
                <form
                    className="row"
                    onSubmit={e => {
                        e.preventDefault()
                        if (!isLoading && !isConversationEnded && input.trim()) sendMessage()
                    }}
                >
                    <textarea
                        className="input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder="請輸入您的問題..."
                        disabled={isLoading || isConversationEnded}
                        aria-label="輸入訊息"
                        rows={3}
                        onCompositionStart={() => setIsComposing(true)}
                        onCompositionEnd={() => setIsComposing(false)}
                        onKeyDown={e => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                // IME 選字中時，不送出
                                if (isComposing || e.nativeEvent?.isComposing || e.keyCode === 229) {
                                    return
                                }
                                e.preventDefault()
                                if (!isLoading && !isConversationEnded && input.trim()) sendMessage()
                            }
                        }}
                    />

                    <button
                        className="btn-end"
                        type="button"
                        disabled={isConversationEnded || isLoading}
                        onClick={endConversation}
                    >
                        結束對話
                    </button>

                    <button className="btn-send" type="submit" disabled={isLoading || isConversationEnded || !input.trim()}>
                        {isLoading ? '傳送中...' : '發送'}
                    </button>
                </form>
            </footer>
        </div>
    )
}
