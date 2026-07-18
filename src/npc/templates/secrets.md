# Secrets — GM-gated knowledge

Clues in THIS file are locked: the NPC sees only the `hint:` lines, never the
text below them, until you approve a reveal. When the players touch one of
these topics the NPC stalls in character and you get a console prompt —
answer `/yes` or `/no`, optionally with a steer, e.g.
`/yes but only vaguely, she is scared` or `/no she lies and blames raiders`.
`/secrets` lists them, `/reveal <id>` unlocks one proactively.

Format: one `## id` per secret (lowercase-with-dashes), a required `hint:`
line (what it concerns — this is all the NPC gets), an optional
`mode:` (`hesitate` = pause audibly, the default; `deflect` = brush it off as
if knowing nothing), then the secret itself. A `revealed:` line is written
back automatically when you approve.

Write each hint in the words the players are likely to use ("the teleporter
key", "what she erased") — the hint is the ONLY thing the NPC can match a
question against, since it never sees the secret itself.

## teleporter-key

hint: the location of a working teleporter key — only for someone with her full trust

The key is hidden in a hollow beneath the shrine's altar stone. It opens the
transit arch on the monolith's north face and still holds three charges.

## erased-discovery

hint: what she removed from the Order's records years ago
mode: deflect

She excised every record of a self-repairing weapon seed found in the
eastern ruins — the same ruins the raiders now camp in. If it wakes, the
region burns; she judged the Order too curious to be trusted with it.
