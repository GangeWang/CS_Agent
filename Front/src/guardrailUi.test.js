import { describe, expect, it } from 'vitest'
import {
    GUARDRAIL_STATUS_COPY,
    INITIAL_GUARDRAIL_UI,
    MAX_EMOTION_LEVEL,
    formatEmotionLevel,
    getNextGuardrailUi,
    normalizeGuardrailLabel,
} from './guardrailUi'

function run(labels) {
    return labels.reduce((ui, label) => getNextGuardrailUi(ui, label), INITIAL_GUARDRAIL_UI)
}

describe('normalizeGuardrailLabel', () => {
    it('把後端的 label 對應到三種狀態', () => {
        expect(normalizeGuardrailLabel('ABUSIVE')).toBe('abuse')
        expect(normalizeGuardrailLabel('PROMPT_ATTACK')).toBe('warning')
        expect(normalizeGuardrailLabel('prompt-attack')).toBe('warning')
        expect(normalizeGuardrailLabel('SPAM')).toBe('warning')
        expect(normalizeGuardrailLabel('NORMAL')).toBe('normal')
        expect(normalizeGuardrailLabel('UNCERTAIN')).toBe('normal')
        expect(normalizeGuardrailLabel(undefined)).toBe('normal')
    })
})

describe('getNextGuardrailUi', () => {
    it('有情緒的訊息累加情緒強度，最多到上限', () => {
        const ui = run(Array(MAX_EMOTION_LEVEL + 3).fill('ABUSIVE'))
        expect(ui.state).toBe('abuse')
        expect(ui.abuseLevel).toBe(MAX_EMOTION_LEVEL)
    })

    it('平穩的訊息逐步降回正常', () => {
        expect(run(['ABUSIVE', 'ABUSIVE', 'NORMAL'])).toMatchObject({ state: 'abuse', abuseLevel: 1 })
        expect(run(['ABUSIVE', 'NORMAL'])).toMatchObject({ state: 'default', abuseLevel: 0 })
    })

    it('高風險請求顯示警示，但不改變情緒強度', () => {
        expect(run(['ABUSIVE', 'PROMPT_ATTACK'])).toMatchObject({ state: 'warning', abuseLevel: 1 })
    })
})

describe('文案', () => {
    it('ABUSIVE 顯示成情緒與安撫，不是不當語氣', () => {
        expect(GUARDRAIL_STATUS_COPY.abuse.text).toContain('安撫')
        expect(GUARDRAIL_STATUS_COPY.abuse.text).not.toContain('不當')
        expect(formatEmotionLevel(3)).toBe(`情緒強度 3/${MAX_EMOTION_LEVEL}`)
    })
})
