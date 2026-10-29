"""Vision contract for visible, enabled table controls; no site-specific geometry."""

TABLE_UI_SCHEMA = {
    'type': 'object',
    'properties': {
        'game': {'type': 'string', 'enum': ['nlhe', 'other', 'unknown']},
        'table_count': {'type': 'integer', 'minimum': 0, 'maximum': 10},
        'amount_mode': {'type': 'string', 'enum': ['raise_to', 'chips_to_add', 'unknown']},
        'amount_units': {'type': 'string', 'enum': ['chips', 'bb', 'unknown']},
        'decimal_separator': {'type': 'string', 'enum': ['dot', 'comma', 'unknown']},
        'amount_value': {'type': 'number', 'minimum': 0},
        'controls': {'type': 'array', 'maxItems': 8, 'items': {
            'type': 'object', 'properties': {
                'role': {'type': 'string', 'enum': ['fold', 'check', 'call', 'bet', 'raise', 'amount']},
                'bounds': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                           'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000}},
                'enabled': {'type': 'boolean'},
                'label': {'type': 'string'},
                'kind': {'type': 'string', 'enum': ['button', 'input', 'unsupported']},
            }, 'required': ['role', 'bounds', 'enabled', 'label', 'kind']}},
    },
    'required': ['game', 'table_count', 'amount_mode', 'amount_units', 'decimal_separator', 'controls'],
}

CONTROL_PROMPT = """
Also return table_ui for control automation. Count poker tables visible in the
capture, including partial tables. If multiple tables or hero identity is
ambiguous, do not guess a target. Identify No-Limit Hold'em from visible
evidence; game=unknown when rules are unclear, other for other games.
Hero may sit anywhere. Identify their own face-up hole cards and controls
from self/you labels and unambiguous ownership, never assume bottom-center.
Return only currently visible controls belonging to this hero on this table.
Never return pre-action checkboxes, auto-fold options, navigation, chat,
buy-in/rebuy, deposits, or controls from another table. enabled=true only for
an enabled immediate action on this hero turn; no guessing from button color.
bounds=[left,top,right,bottom], normalized 0..1000 separately along the entire
input image width/height, NOT the detected table or a cropped subsection.
label=visible text (or visible associated label for the amount field).
kind=input only for a direct editable numeric amount field. Sliders, custom
keypads, unlabeled fields and controls needing extra dialogs are unsupported.
For a shared check/call or bet/raise button, give its CURRENT visible role.
amount_mode=raise_to or chips_to_add ONLY when explicit on-screen evidence
establishes which total the field accepts; otherwise unknown. Never infer
this from a site's usual behavior. amount_units=bb only when the INPUT field
explicitly takes BB units; chips for chip/currency units; otherwise unknown.
decimal_separator from visible numeric formatting, unknown when ambiguous.
amount_value is the currently visible number INSIDE the editable input, in
that input's units; omit it when unreadable. For comma decimals, output the
equivalent JSON number. Do not copy a suggested preset or button amount.
Unknown semantics must remain unknown and prevent automated raising.
"""
