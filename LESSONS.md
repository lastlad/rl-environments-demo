# Lessons from the project

Most of the lessons in this project came from runs that failed. Each section
gives one run, in three parts:

- **What happened:** the result of the run.
- **Cause:** the reason for the result.
- **Lesson:** what to do in a different project.

The numbers are from the original runs. At that time, the exam did not have a
fixed size of 100 tickets. For this reason, the exam sizes are different. The current
scores are in the README.

## 1. A score of 1 of 50 was not a defect of the model

**What happened.** On its first exam, untrained Gemma got a score of 1 of 50.

**Cause.** Gemma writes reasoning before its answer. The test script permitted
only 100 tokens for each answer. For this reason, the script stopped each answer before
the tool call. Then the limit changed to 600 tokens, and the script removed
the reasoning before it read the answer. The same model got 23 of 30.

**Lesson.** When a score is almost zero, read some raw outputs of the model
first. Then make a decision about the model. The code around the model (the
harness) has defects more frequently than the model.

## 2. A real model found the defect in `naive_reward`

**What happened.** Untrained Gemma had 7 failures, and all 7 were the same.
The correct action was to escalate. Gemma told the customer "escalated", but
it did not call the `escalate` tool.

**Cause.** To write the correct words is easier than to do the correct action.
`naive_reward` examines only the words. For this reason, it gives a correct score to these
7 tickets. `verified_reward` examines the order database. It found all 7
failures.

**Lesson.** Give a score to the changes in the environment, not to the text of
the agent. Step 2 shows a small program that learns this incorrect behaviour.
A real model can do it without training.

## 3. Supervised training removed the reasoning of the model

**What happened.** After supervised fine-tuning on the episodes of the expert,
Gemma got 42 of 50. In all 8 failures, the correct action was a refund. Gemma
escalated these tickets.

**Cause.** The examples of the expert contained only tool calls. They did not
contain reasoning. The training taught Gemma to answer immediately, without
reasoning. Without reasoning, Gemma did not compare the amount with the limit.
It used one general habit: escalate all recent orders.

**Lesson.** A model learns all of the properties of your examples. It also
learns from the data that the examples do not contain. Examine how a trained
model fails, and also how frequently it fails.

## 4. Reinforcement fine-tuning made a model that guesses worse

**What happened.** The training started from the weights in lesson 3.
Reinforcement fine-tuning changed the score from 42 to 35 to 29 of 50.

**Cause.** On recent orders, the model guessed between "escalate" and
"refund". Reinforcement fine-tuning keeps the attempts that are correct. Here,
the correct attempts were guesses. Training on them changed the habit from
"escalate all orders" to "refund all orders".

**Lesson.** Reinforcement fine-tuning makes stronger the behaviour that the
model already shows for the correct reason. It cannot find the difference
between a correct guess and a correct decision. For this reason, on a model that
guesses, it makes the guesses stronger.

## 5. A model that was too small did not improve

**What happened.** A much smaller model (Qwen 0.5B) got 21, 21, 13 and 15 of
50 in three rounds. An earlier run used larger training steps. Its score
decreased from 21 to 7.

**Cause.** Most of the correct attempts of the model were guesses. For this reason, there
was no real skill in them to make stronger.

**Lesson.** Test the untrained model first (step 4). If the model is never
correct for the correct reason, use a stronger model or supervised training.
Then use reinforcement training.

## 6. The training measured the incorrect part of each example

**What happened.** The first reinforcement run that kept the reasoning of
Gemma made the score worse: from 23 to 17 of 30. The model stopped its
reasoning again. But each training example contained reasoning.

**Cause.** Each example was the full conversation, and the training measured
all of it. The text format of Gemma removes the reasoning from the earlier
turns of a conversation. For this reason, most of the measured text contained answers
without reasoning. The model learned this pattern.

**Correction.** `agents/local_trainer.py` now measures only the text that the
model wrote at the current step. After this one change, the same run got 23,
26, 30 and 29 of 30.

**Lesson.** Know exactly which part of each example the training measures.

## 7. The conditions for success

Reinforcement fine-tuning was successful when three conditions were true:

- The untrained model already had correct reasoning. It completed most tickets
  correctly.
- Its only fault was that it did not always do one step: the call to `escalate`. Sometimes the
  model did the step. For this reason, there were good attempts to keep.
- The training examples contained the reasoning, and the training measured the
  correct part.

This run did not use an expert. Gemma learned only from the score from
`verified_reward`.

## 8. Small exams hide differences and show false differences

The first results compared 23 of 30, 42 of 50 and 59 of 60. These were three
different exams. With 30 tickets, one ticket is 3 points. For this reason, a change from
30 to 29 is not important. Now all agents do the same 100 tickets
(`refund_env/exam.py`). The exam also gives the score for each ticket type.
This detailed score showed lessons 2, 3 and 4.

## 9. Defects in the environment that we corrected

- **Units.** The rules gave the limit as "$100", but the orders showed
  `amount_cents`. For this reason, the test measured unit conversion, not the refund
  policy. Now the prompt gives the limit in cents. After this correction,
  supervised fine-tuning got 96 of 100. Before the correction, it got 42 of
  50. For this reason, a part of the failure in lesson 3 was a defect of the environment.
- **Tools that accepted incorrect calls.** `escalate` accepted unknown
  arguments, and you could call it two times. A tool that accepts an incorrect
  call hides the mistake of the agent. Now the environment rejects both
  incorrect calls. A test checks each one.

## 10. Reinforcement cannot teach a step that the model never does

**What happened.** On the more difficult environment, untrained Gemma got 78
of 100. It did not call `check_customer` on any ticket. After three rounds of
reinforcement fine-tuning, it got 79. It still did not check the customer.

**Cause.** Reinforcement fine-tuning keeps the attempts that are correct.
Most customers do not have a fraud flag. On these tickets, an attempt without
the check is also correct. For this reason, all 155 kept attempts did not have the check.
The training taught Gemma that the check is not necessary.

On the easy environment, Gemma called `escalate` on some tickets. For this reason, there
were good attempts to keep. On this environment, there were no good attempts.

**Lesson.** Reinforcement makes stronger only the behaviour that the model
shows sometimes. A step that the model never does must come from a different
source. Examples are a training example, an instruction in the prompt, or a
reward for the step.

## 11. A more difficult environment tests the training method

**What happened.** On the easy environment, the two training methods had good
results (96 and 100). On the more difficult environment, supervised training
made Gemma worse (from 78 to 68). Reinforcement training had almost no effect
(from 78 to 79). Claude got 100 on the two environments.

**Cause.** Supervised training copied the steps of the expert, but not the
reasoning. For this reason, Gemma checked each customer, but its decisions were bad
(lesson 3). Reinforcement had no new behaviour to make stronger (lesson 10).

**Lesson.** A perfect score shows that the environment is too easy. It does
not show that the training method is good. Make the environment more
difficult, until the best agent makes some mistakes. Then compare the
training methods.

## 12. Teach the step with reasoning, then use reinforcement

**What happened.** We wrote the examples of the expert again. Each step now
contains the reasoning of the expert, in the reasoning format of Gemma. An
example is "Rule 1: the customer is not flagged. Rule 2: delivered 26 days
ago, within 30...".

Supervised training on these examples changed the score of Gemma from 78 to
99 of 100. Gemma checked each customer and refunded half on all 13 opened
items. It sent no refunds that the policy does not allow. Then reinforcement
training started from these weights and got 100 of 100.

**Cause.** Supervised training taught the step that reinforcement could not
find (lesson 10). The reasoning in the examples kept the reasoning of Gemma.
For this reason, Gemma did not use general habits (lesson 3). After this, Gemma got 59 of
its first 60 training tickets correct. For this reason, reinforcement had many good
attempts to keep, and they were correct for the correct reason (lesson 4).

**Lesson.** Use supervised training to teach the steps that the model never
does, with the reasoning in the examples. Then use reinforcement to improve
the steps that the model already does. Do them in this sequence. The change
from 99 to 100 is only one ticket, so it is not important. Here, supervised
training did almost all of the work.

## 13. A perfect score needs a changed test

**What happened.** After training, Gemma got 99 and 100. But all training
tickets and all exam tickets used the same two numbers: deny after 30 days,
and escalate above 10000 cents.

Step 8 used the same 100 tickets with new rules: 14 days and 5000 cents. The
new rules change the correct answer on 36 tickets. There was no new training.
The two trained versions of Gemma got 100, and all 36 changed tickets were
correct. An expert that uses the old numbers gets 64.

**Cause.** In step 5, the reasoning of the expert always gave the number next
to the rule: "Rule 2: delivered 26 days ago, within 30". Gemma learned this
pattern: compare the order with the number in the rules. On the new exam,
Gemma wrote "more than 14, so I deny". The examples taught a method of
reasoning, not only the answers.

**Lesson.** A high score on tickets that are similar to the training tickets
does not show what the model learned. Change a condition that the agent must
manage, and examine the result. This test changed only the numbers. A new
rule, or the rules in a different sequence, is a better test.
