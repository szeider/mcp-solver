
We have a knowledge base that describes different entities and their characteristics, specifically, whether or not they can fly. Model the problem using Answer Set Programming.

### Here’s what we know:

- Tweety is a bird and is yellow.
- Opus is a bird and is a penguin.
- Woody is a bird and is a woodpecker.
- Penguins and woodpeckers are both types of birds.
- Later, we find out that Woody is injured due to a broken wing.
- There is also an airplane named Polly, and Polly can fly.

### General Rules

- By default, birds can fly.
- Penguins cannot fly (this is an exception).
- Injured birds cannot fly (another exception).
- Anything that can fly is considered mobile.
- All birds have feathers.

### Task

For each of the following entities — Tweety, Opus, Woody, and Polly — determine:

1. Can it fly?  
2. Is it mobile?  
3. Does it have feathers?  
4. For each conclusion, specify the type of reasoning used:
   - Was it based on a default rule, an exception, or a direct fact?

## Output Format

Return a single JSON object on stdout with this schema:

- `satisfiable` (boolean): `true` (the program has an answer set).
- `entities` (object): keys are the lowercase entity names `tweety`, `opus`,
  `woody`, `polly`. Each value is an object with:
  - `can_fly` (boolean)
  - `mobile` (boolean)
  - `has_feathers` (boolean)
  - `fly_reasoning` (string): why the fly conclusion was reached, one of
    `"default"` (default bird-flies rule), `"exception"` (penguin or injured
    exception blocks flight), or `"fact"` (flight stated directly).

Expected result:

```json
{
  "satisfiable": true,
  "entities": {
    "tweety": {"can_fly": true, "mobile": true, "has_feathers": true, "fly_reasoning": "default"},
    "opus": {"can_fly": false, "mobile": false, "has_feathers": true, "fly_reasoning": "exception"},
    "woody": {"can_fly": false, "mobile": false, "has_feathers": true, "fly_reasoning": "exception"},
    "polly": {"can_fly": true, "mobile": true, "has_feathers": false, "fly_reasoning": "fact"}
  }
}
```
