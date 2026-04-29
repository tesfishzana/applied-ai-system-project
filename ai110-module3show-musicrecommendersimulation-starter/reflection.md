# Reflection: Profile Comparisons

This file compares pairs of user profiles to explain what changed between their
outputs and why those changes make sense given the scoring logic.

---

## 1. Weekend Vibes vs. Chill Lofi

**Weekend Vibes** (pop / happy / energy=0.80): top result is Sunrise City (3.98/4.50)
**Chill Lofi** (lofi / chill / energy=0.38 / acoustic): top result is Library Rain (4.47/4.50)

The outputs are completely different — different songs, different genres, different
energy levels. This is expected and shows that the genre + energy pair is doing its
job. Weekend Vibes wants loud, danceable pop; Chill Lofi wants quiet, acoustic lofi.
The system correctly steers each listener in opposite directions.

One interesting contrast: the Chill Lofi profile scored its top result *higher*
(4.47) than Weekend Vibes scored its top result (3.98). That happened because Library
Rain matches all four signals — genre, mood, energy, AND acoustic warmth — while
Sunrise City only matches three (genre, mood, energy; not acoustic). A profile that
is very specific and the catalog happens to satisfy completely will always outscore a
profile where one preference is missing from the catalog.

The practical lesson: if you're building a recommender, catalog coverage matters as
much as scoring logic. The lofi listener was lucky that the catalog has three lofi
songs. A pop listener wanting "indie pop" would have found only one song (Rooftop
Lights) and dropped off a cliff after that.

---

## 2. Deep Intense Rock vs. Conflicting (High Energy + Sad Mood)

**Deep Intense Rock** (rock / intense / energy=0.92): top result is Storm Runner (3.99/4.50)
**Conflicting** (pop / sad / energy=0.90): top result is Gym Hero (2.97/4.50)

These two profiles both ask for very high energy (0.92 and 0.90) but differ in mood.
The rock profile's mood ("intense") exists in the catalog and fires correctly. The
sad-mood profile's mood ("sad") does not exist anywhere in the 18-song catalog, so it
earns zero mood points for every single song.

The consequence shows clearly in the scores: Storm Runner earns 3.99 (genre + mood +
energy); Gym Hero earns only 2.97 (genre + energy, no mood). The sad-mood listener
loses an entire point permanently. Even worse, the top recommendation for someone who
explicitly wants sad music is Gym Hero — a high-energy pump-up pop track. The system
does not know the request is impossible and confidently returns the wrong answer.

The fix would be to detect when a requested mood label has no match in the catalog and
alert the user, rather than silently ignoring the preference. For now, this is the
single biggest gap between what the user expects and what the system delivers.

---

## 3. Acoustic Metal Head vs. Classical Serenity

**Acoustic Metal Head** (metal / angry / energy=0.97 / acoustic): top result Iron Collapse (4.00/4.50)
**Classical Serenity** (classical / peaceful / energy=0.22 / acoustic): top result Moonlit Sonata (4.50/4.50)

Both profiles ask for acoustic music. The outcomes are very different.

Classical Serenity achieves a perfect 4.50 score because the one classical song in the
catalog (Moonlit Sonata) happens to match genre, mood, energy, and acousticness
exactly. The acoustic preference works here because classical music naturally tends to
be acoustic.

The Acoustic Metal Head profile, by contrast, never receives its acoustic bonus at all.
Iron Collapse (the only metal song) has an acousticness of 0.04 — essentially zero.
Metal songs are electric, distorted, and loud; acoustic warmth is not a feature of
the genre. The user wanted something that doesn't exist in the catalog, but instead of
being told this, they simply receive Iron Collapse without the acoustic bonus
(4.00 instead of 4.50) and then see country and blues songs in positions #2 and #3
because those genres happen to have acoustic warmth.

The contrast illustrates a general rule: when a user's preferences are internally
consistent with real music (classical + acoustic = natural fit), the system works well.
When preferences contradict each other (metal + acoustic = rare in practice and absent
from this catalog), the system silently substitutes what it has rather than flagging
the impossibility.

---

## 4. Weekend Vibes Standard vs. Weekend Vibes Energy-Doubled (Weight Experiment)

**Standard weights** (genre=2.0 / energy=1.0): ranking is Sunrise City, Gym Hero, Rooftop Lights
**Energy-doubled weights** (genre=1.0 / energy=2.0): ranking is Sunrise City, Rooftop Lights, Gym Hero

Positions #2 and #3 swapped. Gym Hero (pop, energy=0.93) dropped behind Rooftop
Lights (indie pop, energy=0.76). Why?

The user asked for energy=0.80. Rooftop Lights is 0.04 away from that target; Gym Hero
is 0.13 away. Under standard weights, Gym Hero's genre match (worth 2.0) made this
gap irrelevant — genre was more than twice as valuable as energy. When the energy
weight doubled to 2.0 and genre dropped to 1.0, Rooftop Lights' energy advantage
(0.04 gap vs. 0.13 gap) finally mattered enough to push it ahead.

The question of which ranking is "better" depends on what you think the user actually
wants. If genre is the primary identity ("I am a pop listener first"), the original
weights make sense. If energy is the primary experience ("I want something that feels
like 0.80 energy, whatever the genre"), the doubled-energy weights make sense.

This is exactly the kind of judgment call that separates a thoughtfully tuned
recommender from a default one. Neither set of weights is objectively correct —
they encode a theory about what music preference means.

---

## 5. Summary Table

| Profile pair compared          | Key difference in output        | Root cause                                      |
|--------------------------------|---------------------------------|-------------------------------------------------|
| Weekend Vibes vs. Chill Lofi   | Completely different top 5      | Genre + energy target drive opposite directions |
| Deep Rock vs. Conflicting      | 1.0 point gap in top score      | "sad" mood missing from catalog, silent failure |
| Acoustic Metal vs. Classical   | Acoustic bonus fires vs. never  | Acoustic and classical are consistent; metal is not |
| Standard vs. Energy-Doubled    | Positions #2 and #3 swap        | Energy gap matters more when energy weight doubles |
