# Poridhi Labs Development Guide for MLOps/DataOps Team

**Author: Tahnik Ahmed Raiyan**

**For the teammates:**

---

## Welcome

This guide establishes the standards and practices for developing technical labs at Poridhi. It serves as both a reference for experienced developers and a training resource for new team members.

Our labs teach complex technical concepts to learners with varying backgrounds. The quality of our labs directly impacts learning outcomes. This guide ensures consistency, effectiveness, and professionalism across all lab content.

---

## Table of Contents

1. [Philosophy: Why This Approach](#philosophy-why-this-approach)
2. [The Learning Science Behind Our Methods](#the-learning-science-behind-our-methods)
3. [Professional Standards](#professional-standards)
4. [Lab Structure](#lab-structure)
5. [Writing Style Guide](#writing-style-guide)
6. [Active Learning Techniques](#active-learning-techniques)
7. [Code and Commands](#code-and-commands)
8. [Visual Elements](#visual-elements)
9. [Quality Assurance](#quality-assurance)
10. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
11. [Templates and Examples](#templates-and-examples)
12. [Review Process](#review-process)

---

## Philosophy: Why This Approach

### The Problem with Traditional Labs

Traditional technical labs follow a pattern:

1. Present concept
2. Show complete code
3. Tell student to copy it
4. Move to next concept

This approach fails for several reasons:

- **Illusion of competence**: Students feel they understand because the code works, but they have not internalized the concepts
- **No problem-solving**: Without struggle, neural pathways for retention do not form
- **Passive consumption**: Reading and copying engages different brain systems than creating and debugging
- **Poor transfer**: Students cannot apply concepts to new situations because they never truly understood them

### Our Approach

We design labs that require active participation:

- Students predict outcomes before seeing results
- Code is scaffolded with blanks to complete, not copied wholesale
- Experiments deliberately break things to demonstrate why patterns matter
- Self-assessment verifies understanding, not just task completion

This approach requires more effort from students. That effort is precisely what creates learning.

---

## The Learning Science Behind Our Methods

### Retrieval Practice

When students must recall information rather than simply recognize it, memory strengthens. Fill-in-the-blank exercises force retrieval.

**Example**: Instead of showing `status_code=201`, we ask: "What status code indicates resource creation?"

### Desirable Difficulty

Learning that feels easy often does not last. Introducing appropriate challenges improves long-term retention.

**Example**: Predicting output before running a command creates uncertainty. Resolving that uncertainty reinforces the concept.

### Elaborative Interrogation

Asking "why" questions forces deeper processing than "what" questions.

**Example**: "Why do we use HTTPException instead of returning None?" requires understanding, not just recognition.

### Spaced Verification

Checking understanding at multiple points, rather than only at the end, identifies gaps early.

**Example**: Self-assessment checklists after each chapter, plus final conceptual questions.

---

## Professional Standards

Our labs represent Poridhi's technical brand. They must meet enterprise standards.

### Language Requirements

| Category | Prohibited | Preferred |
|----------|------------|-----------|
| Emojis | All emojis (including checkmarks, rockets, etc.) | None |
| Exclamations | "This is amazing!" | "This pattern provides..." |
| Casual phrases | "Let's dive in", "Pretty cool" | Direct statements |
| Clichés | "Magic happens", "Under the hood", "Secret sauce" | Technical descriptions |
| Filler | "Basically", "Actually", "Simply" | Remove entirely |

### Tone

- **Authoritative but not condescending**: Assume intelligence, not prior knowledge
- **Direct**: State what to do, not what we will do
- **Precise**: Use exact technical terms
- **Neutral**: Avoid subjective assessments of difficulty

**Avoid:**
> "Don't worry, this is actually pretty simple once you get the hang of it!"

**Prefer:**
> "This pattern is common in production systems. The following sections explain each component."

### Pronouns

- Use "you" when addressing the student
- Avoid "we" (creates false intimacy)
- Avoid "I" (labs are not personal narratives)

**Avoid:**
> "Let's create our first endpoint together."

**Prefer:**
> "Create the first endpoint by adding the following code."

Or, for active learning:
> "Complete the endpoint definition below."

---

## Lab Structure

### Required Sections

Every lab must include:

```
# [Lab Title]

## Introduction
## Learning Objectives
## Prologue: The Challenge
## Environment Setup
## Chapter 1-N: [Content Chapters]
## Epilogue: The Complete System
## The Principles
## Troubleshooting
## Next Steps
## Additional Resources
```

### Section Details

#### Introduction

Two to three sentences explaining:
- What the lab teaches
- How this lab approaches the topic (if distinct from typical tutorials)
- Target audience (implicit through prerequisites)

Include an architecture diagram if the lab involves system design.

#### Learning Objectives

Numbered list of measurable outcomes. Use action verbs:

| Strong Verbs | Weak Verbs |
|--------------|------------|
| Create, Implement, Configure | Understand, Learn, Know |
| Debug, Analyze, Evaluate | See, Explore, Discover |
| Design, Optimize, Integrate | Appreciate, Recognize |

**Example:**
```markdown
By the end of this lab, you will be able to:

1. Create a FastAPI application with health and readiness checks
2. Implement input validation using Pydantic models
3. Configure lifespan hooks for resource management
4. Debug common API errors using status codes
```

State prerequisites clearly:
```markdown
**Prerequisites:** Basic Python knowledge and familiarity with REST API concepts.
```

#### Prologue: The Challenge

Scenario-based context that motivates the lab content. Elements:

- Role assignment ("You join a team...", "You inherit a system...")
- Problem statement (what is not working, what is needed)
- Success criteria (what the student will build)

This section answers: "Why does this matter in the real world?"

**Example:**
```markdown
## Prologue: The Challenge

You join the platform safety team at a social media company. User comments 
require automated analysis: sentiment detection for potential harassment, 
trust scoring based on user reputation metrics, and robust validation to 
prevent abuse.

Currently, moderators review everything manually. The backlog grows daily. 
Your task is to build a Content Moderation API that handles these analyses 
automatically.
```

#### Environment Setup

Provide exact commands. Do not assume environment configuration.

Required elements:
- System updates
- Virtual environment creation
- Package installation with versions if critical
- Project structure creation

**Example:**
```markdown
## Environment Setup

Update your system and install required packages:

```bash
sudo apt update
sudo apt install -y python3-venv
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn pydantic
```

Create the project structure:

```bash
mkdir -p src tests
```
```

#### Chapters

See [Chapter Template](#chapter-template) below.

#### Epilogue: The Complete System

Summary of what was built. Include:

- Endpoint/feature table
- End-to-end verification commands (all in sequence)
- Reference to auto-generated documentation if applicable

**Example:**
```markdown
## Epilogue: The Complete System

Your single application now provides:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check |
| POST | `/predict` | Sentiment prediction |

Verify all endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test input"}'
```
```

#### The Principles

Numbered list of generalizable takeaways. These should apply beyond the specific lab content.

**Example:**
```markdown
## The Principles

1. **Start with operational endpoints** — Health and readiness checks come first
2. **Validate at the boundary** — Reject invalid input before it reaches business logic
3. **Manage resources efficiently** — Load expensive resources once, not per-request
4. **Report failures honestly** — Use appropriate status codes for different error types
```

#### Troubleshooting

Common errors with solutions. Format:

```markdown
## Troubleshooting

### Error: Address already in use

**Cause:** Another process is using port 8000.

**Solution:**
```bash
lsof -ti:8000 | xargs kill -9
```

### Error: ModuleNotFoundError

**Cause:** Virtual environment not activated or packages not installed.

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```
```

#### Next Steps

Suggested extensions for students who complete the lab. These should be challenging but achievable based on lab content.

#### Additional Resources

Official documentation links only. No blog posts, no outdated tutorials.

---

## Chapter Template

Each chapter follows this structure:

### Opening Context (2-3 sentences)

Explain WHY this chapter matters. Connect to the overall problem.

**Example:**
```markdown
## Chapter 3: Input Validation

The registration system accepts user input. Without validation, malformed 
data could corrupt the database or cause downstream errors. Pydantic 
provides automatic validation that rejects invalid requests before they 
reach business logic.
```

### N.1 What You Will Build

Brief description of the feature and key concepts.

### N.2 Think First: [Topic]

Pre-implementation questions that activate prior knowledge and create curiosity.

Use `<details>` tags for answers:

```markdown
### 3.2 Think First: Validation Scenarios

Consider this incoming request:

```json
{"model_id": "not-a-number", "accuracy": 1.5}
```

**Question:** Identify at least two problems with this data.

<details>
<summary>Click to review</summary>

1. `model_id` should be an integer, not a string
2. `accuracy` exceeds the valid range of 0.0 to 1.0

</details>
```

### N.3 Implementation

Scaffolded code with blanks. Provide hints. Complete solution in collapsible section.

### N.4 Understanding the Code

Matching exercises, concept questions, or technical explanations.

### N.5 Test and Verify

Prediction exercises before running tests. Expected outputs in collapsible sections.

### N.6 Checkpoint

Self-assessment checklist:

```markdown
**Self-Assessment:**
- [ ] Endpoint returns expected response
- [ ] Invalid input is rejected with 422 status
- [ ] You can explain why we use Field() constraints
```

### N.7 Experiment (Optional)

Deliberate failure scenarios that demonstrate why patterns matter.

---

## Writing Style Guide

### Sentence Structure

- Lead with the action or key concept
- Keep sentences under 25 words when possible
- One idea per sentence

**Avoid:**
> "What we're going to do now is we're going to add a new endpoint that will handle the registration of new models in our system."

**Prefer:**
> "Add a registration endpoint for new models."

### Paragraph Structure

- First sentence states the main point
- Supporting sentences provide details
- Maximum 4-5 sentences per paragraph
- Single-line paragraphs are acceptable for emphasis

### Technical Terms

- Define on first use if not in prerequisites
- Use consistent terminology throughout
- Prefer industry-standard terms over invented ones

### Transitions Between Sections

Connect chapters with brief context, not meta-commentary.

**Avoid:**
> "Now let's move on to the next section where we'll learn about..."

**Prefer:**
> "The catalog displays existing models. Adding new models requires a registration endpoint."

---

## Active Learning Techniques

### 1. Fill-in-the-Blanks

Replace complete code with scaffolds:

```python
@app.post("/users", status_code=___)  # Q1: What code for "created"?
async def create_user(user: ___):      # Q2: What validates input?
    pass
```

**Guidelines:**
- 2-4 blanks per code block (more becomes frustrating)
- Number questions for easy reference
- Provide hints if concepts are new
- Always include complete solution in `<details>`

### 2. Prediction Exercises

Before any test command:

```markdown
**Predict:** What status code will this return?

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": ""}'
```

<details>
<summary>Click to verify</summary>

**422 Unprocessable Entity** — The empty name violates the `min_length=1` constraint.

</details>
```

### 3. Matching Exercises

For concept verification:

```markdown
Match each status code to its meaning:

| Code | Meaning (A-E) |
|------|---------------|
| 200 | ___ |
| 201 | ___ |
| 400 | ___ |
| 404 | ___ |
| 422 | ___ |

**Options:**
- A: Resource created
- B: Request successful
- C: Validation error
- D: Resource not found
- E: Bad request (business logic)
```

### 4. Scenario Questions

Present realistic situations:

```markdown
**Scenario:** A load balancer health check returns 200, but users report 
the API is not responding to predictions.

**Question:** What endpoint should the load balancer check instead? Why?

<details>
<summary>Click to review</summary>

The load balancer should check `/ready` instead of `/health`. The health 
endpoint only confirms the process is running. The readiness endpoint 
confirms the model is loaded and the service can fulfill requests.

</details>
```

### 5. Deliberate Failure Experiments

Guide students to break things:

```markdown
### Experiment: Validation Bypass

1. In your Pydantic model, remove the `ge=0.0, le=1.0` constraint from accuracy
2. Restart the server
3. Register a model with `accuracy: 150`
4. Query the model list

**Observe:** The invalid data is now in your database.

**Question:** In a production system, what downstream problems could this cause?

5. Restore the constraint and restart
```

### 6. Self-Assessment Checklists

End each chapter:

```markdown
**Self-Assessment:**
- [ ] Server starts without errors
- [ ] All test commands produce expected output
- [ ] You can explain why readiness differs from health
- [ ] You can predict what happens with invalid input
```

---

## Code and Commands

### Code Blocks

**Incomplete (exercises):**
```python
class User(BaseModel):
    name: str = Field(..., ___=1)  # Minimum length constraint
    email: ___                      # Type for email field
```

**Complete (solutions):**
```python
class User(BaseModel):
    name: str = Field(..., min_length=1)
    email: str
```

**Guidelines:**
- Use triple backticks with language identifier
- Include comments for non-obvious lines
- Break long lines for readability
- Use consistent naming conventions

### Command Blocks

Always include expected output or wrap in prediction:

```bash
curl http://localhost:8000/health
```

**Expected output:**
```json
{"status": "healthy"}
```

For destructive commands, add warnings:

```bash
# WARNING: This kills all processes on port 8000
lsof -ti:8000 | xargs kill -9
```

### File References

When students should create files:

```markdown
Create `app.py`:

```python
# app.py
from fastapi import FastAPI
...
```
```

When adding to existing files:

```markdown
Add the following to `app.py` after the health endpoint:

```python
@app.get("/ready")
...
```
```

---

## Visual Elements

### Diagrams

- Use existing diagrams from the asset repository when available
- Reference with full URLs
- Add contextual notes if diagram differs from lab configuration

```markdown
![Architecture Overview](https://raw.githubusercontent.com/.../diagram.svg)

> **Note:** Diagram shows three services. This lab implements only the API service.
```

### Tables

Use tables for:

| Use Case | Example |
|----------|---------|
| Status code references | 200, 201, 400, 404, 422 meanings |
| Endpoint summaries | Method, path, description |
| Concept comparisons | Approach A vs Approach B |
| Matching exercises | Code element to purpose |

### Screenshots

- Include only when UI interaction is essential
- Add notes if screenshot shows different configuration
- Crop to relevant area

```markdown
![MLflow Dashboard](https://github.com/.../screenshot.png)

> **Note:** Screenshot shows port 5000. Your configuration may differ.
```

---

## Quality Assurance

### Pre-Publish Checklist

Before submitting a lab for review:

**Content Quality:**
- [ ] All code has been tested and works
- [ ] Commands produce documented output
- [ ] No broken image links
- [ ] Prerequisites accurately reflect requirements

**Active Learning:**
- [ ] Every chapter has a "Think First" section
- [ ] Code blocks include fill-in-the-blanks where appropriate
- [ ] Tests include prediction exercises
- [ ] Solutions are in collapsible sections
- [ ] Self-assessment at each chapter end
- [ ] Final conceptual questions at lab end
- [ ] At least one experiment section per lab

**Professional Standards:**
- [ ] No emojis
- [ ] No cliché phrases
- [ ] No exclamation marks in technical content
- [ ] Consistent terminology throughout
- [ ] All links functional

**Structure:**
- [ ] All required sections present
- [ ] Troubleshooting covers common errors
- [ ] Next steps provide meaningful extensions

### Ratio Check

Evaluate your lab's active/passive ratio:

| Element | Classification |
|---------|----------------|
| Explanatory paragraphs | Passive |
| Complete code to copy | Passive |
| Fill-in-the-blank code | Active |
| Prediction exercises | Active |
| Matching exercises | Active |
| Experiments | Active |
| Self-assessment | Active |

Target: 70% active, 30% passive.

---

## Common Mistakes to Avoid

### 1. The Code Dump

**Problem:** Presenting 50+ lines of code at once.

**Solution:** Break into logical chunks. Introduce each chunk with context. Use progressive building.

### 2. Missing Context

**Problem:** Jumping to implementation without explaining purpose.

**Solution:** Every chapter opens with WHY before HOW.

### 3. Assumed Knowledge

**Problem:** Using terms or concepts without introduction.

**Solution:** Either explain on first use or list as prerequisite.

### 4. Passive Commands

**Problem:** "Run this command" with no engagement.

**Solution:** Add prediction, or explain what to observe in the output.

### 5. Missing Error Cases

**Problem:** Only showing the happy path.

**Solution:** Include validation tests, error scenarios, and edge cases.

### 6. Outdated Information

**Problem:** Using deprecated APIs or old syntax.

**Solution:** Verify against current documentation. Date labs with version requirements.

### 7. Inconsistent Naming

**Problem:** `app.py` in one section, `main.py` in another.

**Solution:** Use consistent names throughout. Create naming conventions document if needed.

---

## Templates and Examples

### Chapter Opening

```markdown
## Chapter N: [Descriptive Title]

[Context sentence explaining what challenge this addresses.]
[Second sentence connecting to overall lab goal.]
[Optional third sentence previewing the approach.]
```

### Think First Section

```markdown
### N.2 Think First: [Topic]

[Setup scenario or question context]

**Question:** [Specific question requiring thought]

<details>
<summary>Click to review</summary>

[Answer with explanation of reasoning]

</details>
```

### Fill-in-the-Blank Exercise

```markdown
### N.3 Implementation

Complete the following code:

```python
# Code with ___ for blanks and # QN: hints
```

**Hints:**
- Hint for Q1
- Hint for Q2

<details>
<summary>Click to see solution</summary>

```python
# Complete code
```

**Answers:**
- Q1: Explanation
- Q2: Explanation

</details>
```

### Checkpoint

```markdown
### N.6 Checkpoint

**Predict:** [What to predict about the test]

```bash
# Test command
```

<details>
<summary>Click to verify</summary>

**Expected output:**
```json
{"key": "value"}
```

</details>

**Self-Assessment:**
- [ ] Checklist item 1
- [ ] Checklist item 2
- [ ] Checklist item 3
```

### Before/After Example

**Before (Passive):**
```markdown
Add this code to create the health endpoint:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

Run the server and test it:

```bash
curl http://localhost:8000/health
```
```

**After (Active):**
```markdown
### Think First

**Question:** What HTTP method should a health check use? Why?

<details>
<summary>Click to review</summary>

GET — Health checks retrieve status information without modifying state. 
GET is idempotent and cacheable, making it appropriate for status endpoints.

</details>

### Implementation

Complete the health check endpoint:

```python
@app.___("/health")  # Q1: What HTTP method?
async def health_check():
    return {"status": "___"}  # Q2: What status indicates success?
```

<details>
<summary>Click to see solution</summary>

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

</details>

### Checkpoint

**Predict:** What will this command return?

```bash
curl http://localhost:8000/health
```

<details>
<summary>Click to verify</summary>

```json
{"status": "healthy"}
```

</details>
```

---

## Review Process

### Self-Review

Before submitting:

1. Run through the entire lab as a student would
2. Verify all code works
3. Check all links
4. Apply the quality checklist

### Peer Review

Reviewers should check:

1. **Accuracy**: Does the code work? Are explanations correct?
2. **Clarity**: Would a student with stated prerequisites understand?
3. **Engagement**: Is the active/passive ratio appropriate?
4. **Standards**: Does it meet professional guidelines?

### Feedback Guidelines

When reviewing others' work:

- Be specific: "Line 45 uses deprecated syntax" not "Some code is outdated"
- Be actionable: Suggest fixes, not just problems
- Be constructive: Focus on improvement, not criticism

### Revision Cycle

1. Author submits draft
2. Reviewer provides feedback (specific, actionable)
3. Author revises
4. Final review
5. Publication

---

## Quick Reference Card

### Required Sections
Introduction | Objectives | Prologue | Setup | Chapters | Epilogue | Principles | Troubleshooting | Next Steps | Resources

### Chapter Structure
Context | What You Will Build | Think First | Implementation | Understanding | Test | Checkpoint | Experiment

### Active Techniques
Fill-in-blanks | Predictions | Matching | Scenarios | Experiments | Self-assessment

### Prohibited
Emojis | Clichés | Exclamations | "We/Let's" | Filler words

### Target Ratio
70% Active | 30% Passive

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-05 | Lab Development Team | Initial guide |

---

## Questions and Feedback

If you encounter situations not covered by this guide, or have suggestions for improvement, raise them with the Lab Development Lead. This guide evolves based on team experience and student feedback.

For urgent questions during lab development, consult with a senior team member before deviating from these standards.

