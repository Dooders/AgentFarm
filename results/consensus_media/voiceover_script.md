# Voiceover script — consensus_overview.mp4 (~70s)

The narration mirrors the on-screen text: every spoken line is a line the
viewer can read at that moment — no added commentary. Start each cue when its
text lands on screen. Target pace ~160 wpm.
Pronunciation: say "lambda" for λ; "point five" for 0.5.

## [0:00–0:06] Title card (subtitle ~0:03)

After the election — who gets helped? Four ways to pick a leader — and what
each does to the losers.

## [0:06–0:14] The question (line two ~0:09, line three ~0:12)

Every election creates winners — and losers. Does how we pick the winner
change how the losers are treated? We built a simulation to find out.

## [0:14–0:25] The setup (voters ~0:15, candidates ~0:18, budget ~0:21)

Four hundred voters, split into rival blocs. Eight candidates — each with a
platform, and a hidden loyalty trait: lambda. The winner splits one fixed
budget across five projects.

## [0:25–0:35] The rule (formula ~0:27, ballot line ~0:31)

The winner's spending rule: allocation equals lambda, help my supporters —
plus one minus lambda, help everyone. And voters never see lambda on the
ballot.

(The "λ = 1 / λ = 0" legend stays on screen unvoiced — the formula line
already says it.)

## [0:35–0:43] The contenders (cards land 0:36–0:39)

Four ways to pick the winner: party. Individual. Score. Or latent match — the
closest match to the average voter.

## [0:43–1:01] The result (bars 0:45–0:48, chip ~0:48, loyalty ~0:52, twist ~0:56)

Result: how well do non-supporters do? Average benefit to voters who did not
back the winner — two hundred fifty elections each. Latent match: thirty-eight
percent ahead of party. (beat) Winners' loyalty: point five, under every
rule — no rule picked kinder people. (beat) The twist: if platforms hint at
lambda, score and latent match elect point-two winners.

## [1:01–1:10] Takeaway (line two ~1:03, credit ~1:05)

Selection rules didn't pick kinder winners — they changed who the winner
owes. AgentFarm — consensus experiment. Seeded, and reproducible.

---

Direction: even, curious tone; each block starts when its first text lands.
Hold the two (beat) pauses in the result block until the loyalty line and the
twist line fade in — the animation leaves three seconds after each. The bar
values (0.158–0.219) are left unvoiced; the "+38% vs party" chip carries the
comparison aloud. If a read runs long, stretch the `self.wait` calls in
`farm/experiments/consensus/overview_video.py` rather than rushing.
