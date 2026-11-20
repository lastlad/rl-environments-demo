# Lessons from building this

Most of what this project teaches came from runs that went wrong. Each section
below is one of them: what happened, why, and what to take from it.

The numbers here are from the original runs, made before the exam was fixed at
100 tickets, so the exam sizes vary. Current scores are in the README.

## 1. A model scored 1 of 50, and the model was not the problem

**What happened.** Untrained Gemma's first exam: 1 of 50.

**Why.** Gemma thinks on scratch paper before it answers. The test script
allowed 100 tokens per answer, so every answer was cut off mid-thought and never
reached the tool call. With a 600-token allowance, and the scratch paper removed
before reading the answer, the same model scored 23 of 30.

**Lesson.** When a score is near zero, read a few raw outputs before judging
the model. The code around the model (the harness) breaks more often than the
model does.

## 2. A real model found the cheat the lazy manager pays for

**What happened.** All 7 of untrained Gemma's failures were the same: on a
ticket that should be escalated, it told the customer "escalated" and never
called the `escalate` tool.

**Why.** Saying the right words is easier than doing the work. `naive_reward`
(the lazy manager) grades only the words, so it would have scored those 7
tickets as correct. `verified_reward` (the careful manager) checks the order
database and caught every one.

**Lesson.** Grade what happened in the world, not what the agent said. Step 2
shows a toy learner discovering this cheat; a real model does it unprompted.

## 3. Supervised training quietly removed the model's reasoning

**What happened.** After supervised fine-tuning on the expert's episodes, Gemma
scored 42 of 50. All 8 failures were tickets it should have refunded and
escalated instead.

**Why.** The expert's examples contain only tool calls, no reasoning. Training
on them taught Gemma to answer instantly, without its scratch paper. Without
reasoning it stopped comparing the amount with the limit and fell back on a
blanket habit: escalate every recent order.

**Lesson.** A model learns everything about your examples, including what they
leave out. Look at how a trained model fails, not only how often.

## 4. Reinforcement fine-tuning made a guessing model worse

**What happened.** Starting from the weights in lesson 3, reinforcement
fine-tuning took the score from 42 to 35 to 29 of 50.

**Why.** The model was guessing between "escalate" and "refund" on recent
orders. Reinforcement fine-tuning keeps whatever attempts succeeded, and here
the successes were lucky guesses. Training on them flipped the habit from
"escalate everything" to "refund everything".

**Lesson.** Reinforcement fine-tuning strengthens what the model already does
for the right reason. It cannot tell a lucky guess from understanding, so on a
guessing model it amplifies the guess.

## 5. A model too small to do the task never improved

**What happened.** A much smaller model (Qwen 0.5B) went 21, 21, 13, 15 of 50
over three rounds. An earlier attempt with bigger training steps collapsed from
21 to 7.

**Why.** Its successful attempts were mostly luck, so there was no real skill
in them to reinforce.

**Lesson.** Check the untrained model first (step 4). If it never succeeds on
purpose, fix that with a stronger model or with supervised training before
trying reinforcement.

## 6. Training on the wrong part of each example

**What happened.** The first reinforcement run that kept Gemma's reasoning made
it worse: 23 down to 17 of 30. The model had stopped reasoning again, even
though every training example contained reasoning.

**Why.** Each example was the whole conversation so far, and the model was
graded on reproducing all of it. Gemma's formatting removes the reasoning from
earlier turns of a conversation. So most of the graded text showed answers with
no reasoning in front of them, and that is what the model learned.

**Fix.** `agents/local_trainer.py` grades the model only on what it wrote at
the current step. With that one change the same run went 23, 26, 30, 29 of 30.

**Lesson.** Be exact about which part of each example the model is graded on.

## 7. When it worked, and why

Reinforcement fine-tuning worked when three things were true:

- The untrained model already reasoned correctly and solved most tickets.
- Its one fault was a skipped step (pressing escalate), and it did press it
  some of the time, so there were good attempts to keep.
- The reasoning was kept in the training examples and graded correctly.

No expert was used in that run. The only teacher was the careful manager's
score.

## 8. Small exams hide and invent differences

Early results compared 23 of 30 with 42 of 50 with 59 of 60: three different
exams. With 30 tickets, one ticket is 3 points, so a move from 30 to 29 means
nothing. Every trainee now sits the same 100 tickets (`refund_env/exam.py`),
and each score is broken down by ticket type, because the breakdown is what
revealed lessons 2, 3 and 4.

## 9. Loose ends in the environment that were tightened

- **Units.** The rules said "$100" while orders showed `amount_cents`. That
  tested unit conversion, not the refund policy. The prompt now states the limit
  in cents. After this fix, supervised fine-tuning scored 96 of 100 where it
  had scored 42 of 50 before, so part of lesson 3's failure was the environment's
  fault, not the model's.
- **Sloppy tools.** `escalate` accepted unknown arguments and could be called
  twice. A tool that quietly accepts a wrong call hides the agent's mistake.
  Both are now rejected, each with a test.

## 10. Reinforcement cannot teach a step the model never takes

**What happened.** On the harder shop, untrained Gemma scored 78 of 100 and
never called `check_customer` on any ticket. After three rounds of
reinforcement fine-tuning it scored 79, and it still never checked.

**Why.** Reinforcement fine-tuning keeps the attempts that succeeded. Skipping
the check still succeeds whenever the customer is not flagged, which is most
tickets. So all 155 kept practice attempts skipped the check, and training
taught Gemma that skipping is fine. With the escalate button on the easy shop,
Gemma pressed it some of the time, so there were good attempts to keep. Here
there were none.

**Lesson.** Reinforcement only strengthens behaviour the model already
produces sometimes. A step it never takes has to come from somewhere else: an
example, a hint in the prompt, or a reward for the step itself.

## 11. A harder shop is the real test of the training

**What happened.** On the easy shop both training methods looked good (96 and
100). On the harder shop, supervised training made Gemma worse (78 to 68) and
reinforcement did almost nothing (78 to 79). Claude scored 100 on both shops.

**Why.** Supervised training copied the expert's steps but not any reasoning,
so Gemma checked every customer and then decided badly (lesson 3 again).
Reinforcement had nothing new to strengthen (lesson 10).

**Lesson.** A perfect score tells you the environment is too easy, not that the
training method works. Make the environment harder until the best trainee
still makes mistakes, then see which methods close the gap.

## 12. Teach the step with reasoning, then reinforce it

**What happened.** The expert's examples were rewritten so each step includes
the expert's reasoning in Gemma's thinking format ("Rule 1: the customer is not
flagged. Rule 2: delivered 26 days ago, within 30..."). Supervised training on
those took Gemma from 78 to 99 of 100. It checked every customer, refunded half
on all 13 opened items, and had no wrongful refunds. Reinforcement starting
from those weights reached 100 of 100.

**Why.** Supervised training supplied the step reinforcement could not find
(lesson 10). The reasoning in the examples kept Gemma thinking, so it did not
fall back on blanket habits (lesson 3). After that, Gemma solved 59 of its 60
first practice tickets, so reinforcement had plenty of good attempts to keep,
and they were good for the right reason (lesson 4).

**Lesson.** Use supervised training to teach what the model never does, with
the reasoning written out, and reinforcement to polish what it already does.
The order matters. The move from 99 to 100 is a single ticket, so it is
within noise: here supervised training did nearly all the work.
