// 狀態列：依後端每則訊息回傳的 guardrail label 更新。
// 系統目的是「客人有情緒就安撫」，所以 ABUSIVE（有情緒，不論有沒有髒話）顯示成情緒與安撫，
// 不再顯示成「不當語氣」；提示詞攻擊與廣告（PROMPT_ATTACK / SPAM）才是高風險請求。
// 純函式獨立成一個檔案，方便用 vitest 測試（src/guardrailUi.test.js）。

export const MAX_EMOTION_LEVEL = 5

export const GUARDRAIL_STATUS_COPY = {
    default: { text: 'Guardrail：正常', icon: '✅' },
    abuse: { text: 'Guardrail：偵測到情緒，已切換安撫模式', icon: '💗' },
    warning: { text: 'Guardrail：偵測到高風險請求', icon: '⚠️' },
}

export const INITIAL_GUARDRAIL_UI = { state: 'default', abuseLevel: 0, label: 'normal' }

export function normalizeGuardrailLabel(label) {
    if (typeof label !== 'string') return 'normal'
    const normalized = label.trim().toLowerCase().replace(/[\s-]+/g, '_')
    if (normalized === 'abuse' || normalized === 'abusive') return 'abuse'
    if (normalized === 'prompt_attack' || normalized === 'spam') return 'warning'
    return 'normal'
}

// 有情緒的訊息每次 +1（最多 5），平穩的訊息每次 -1；高風險請求不影響情緒強度。
export function getNextGuardrailUi(prev, label) {
    const normalizedLabel = normalizeGuardrailLabel(label)

    if (normalizedLabel === 'abuse') {
        return {
            state: 'abuse',
            abuseLevel: Math.min(MAX_EMOTION_LEVEL, prev.abuseLevel + 1),
            label,
        }
    }

    if (normalizedLabel === 'warning') {
        return {
            state: 'warning',
            abuseLevel: prev.abuseLevel,
            label,
        }
    }

    const abuseLevel = Math.max(0, prev.abuseLevel - 1)
    return {
        state: abuseLevel > 0 ? 'abuse' : 'default',
        abuseLevel,
        label: 'normal',
    }
}

export function formatEmotionLevel(level) {
    return `情緒強度 ${level}/${MAX_EMOTION_LEVEL}`
}
