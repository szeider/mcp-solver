## University Schedule

---

**User**

> A university needs to schedule 8 courses across 4 time slots on 3 days (Mon/Wed/Fri). Each course needs 2 sessions per week. Constraints:
>
> 1. Three professors with specific course assignments:
>    - Smith: AI, ML, Networks
>    - Jones: Databases, Security
>    - Brown: Physics, Chemistry, Biology
> 2. Professors can't teach overlapping sessions
> 3. Each course needs a specific lab room
>    - 2 CS labs and 1 science lab available
>    - CS courses need CS labs
>    - Science courses need science lab
> 4. Student groups have required course combinations:
>    - CS majors: AI+ML+Databases
>    - Security track: Networks+Security
>    - Pre-med: Physics+Chemistry+Biology
> 5. No student should have class conflicts

**LLM**

> Let me create a MiniZinc model for this scheduling problem.

**MCP Solver:** **`submit_model`**

> ```
> % Constants
> int: n_courses = 8;  
> int: n_slots = 4;
> int: n_days = 3;
> int: n_sessions = 2;
> 
> % Decision variables - schedule[course, session] gives day and slot 
> array[1..n_courses, 1..n_sessions] of var 1..n_days: course_day;
> array[1..n_courses, 1..n_sessions] of var 1..n_slots: course_slot;
> 
> % Course IDs: 1=AI, 2=ML, 3=Networks, 4=Databases, 5=Security, 
> %             6=Physics, 7=Chemistry, 8=Biology
> 
> % Ensure sessions are on different days
> constraint forall(c in 1..n_courses)(
> course_day[c,1] != course_day[c,2]
> );
> 
> % Same professor can't teach overlapping sessions  
> % Smith: courses 1,2,3
> constraint forall(i,j in 1..3, s1,s2 in 1..n_sessions where i < j)(
> (course_day[i,s1] = course_day[j,s2]) ->
> (course_slot[i,s1] != course_slot[j,s2]) 
> );
> % Jones: courses 4,5
> constraint forall(i,j in 4..5, s1,s2 in 1..n_sessions where i < j)(
> (course_day[i,s1] = course_day[j,s2]) ->
> (course_slot[i,s1] != course_slot[j,s2])
> );
> % Brown: courses 6,7,8  
> constraint forall(i,j in 6..8, s1,s2 in 1..n_sessions where i < j)(
> (course_day[i,s1] = course_day[j,s2]) -> 
> (course_slot[i,s1] != course_slot[j,s2])
> );
> 
> % Lab constraints - max 2 CS (1-5) and 1 science (6-8) per slot
> constraint forall(d in 1..n_days, t in 1..n_slots)(
> sum(c in 1..5, s in 1..n_sessions)( 
>  bool2int(course_day[c,s] = d /\\ course_slot[c,s] = t)
> ) <= 2 
> );
> constraint forall(d in 1..n_days, t in 1..n_slots)(
> sum(c in 6..8, s in 1..n_sessions)(
>  bool2int(course_day[c,s] = d /\\ course_slot[c,s] = t) 
> ) <= 1
> );
> 
> % Student group constraints
> % CS Majors (1,2,4 can't overlap)
> constraint forall(i,j in [1,2,4], s1,s2 in 1..n_sessions where i < j)(
> (course_day[i,s1] = course_day[j,s2]) ->
> (course_slot[i,s1] != course_slot[j,s2]) 
> );
> 
> % Security track (3,5 can't overlap)
> constraint forall(s1,s2 in 1..n_sessions)(
> (course_day[3,s1] = course_day[5,s2]) ->
> (course_slot[3,s1] != course_slot[5,s2])
> );
> 
> % Pre-med (6,7,8 can't overlap) 
> constraint forall(i,j in 6..8, s1,s2 in 1..n_sessions where i < j)(
> (course_day[i,s1] = course_day[j,s2]) ->
> (course_slot[i,s1] != course_slot[j,s2]) 
> );
> ```

`Success: Model validated, and parameters initialized dynamically.`

**MCP Solver:** **`solve_model`**

> ```
> {'status': 'SUCCESS', 
> 'solution': Solution(course_day=[[3, 2], [2, 3], [1, 3], [2, 3], [3, 2], [1, 2], [2, 1], [2, 3]], 
>                    course_slot=[[2, 1], [4, 4], [4, 1], [2, 3], [4, 4], [3, 2], [3, 4], [4, 4]]),
> 'solve_time': 0.005}
> ```

**LLM**

> Here is the course schedule formatted as a table:
>
> | Course   | Session 1  | Session 2  |
> | -------- | ---------- | ---------- |
> | AI       | Fri Slot 2 | Wed Slot 1 |
> | ML       | Wed Slot 4 | Fri Slot 4 |
> | Networks | Mon Slot 4 | Fri Slot 1 |
> | Database | Wed Slot 2 | Fri Slot 3 |
> | Security | Fri Slot 4 | Wed Slot 4 |
> | Physics  | Mon Slot 3 | Wed Slot 2 |
> | Chem     | Wed Slot 3 | Mon Slot 4 |
> | Biology  | Wed Slot 4 | Fri Slot 4 |
>
> The schedule satisfies all constraints.


**User**

> Verify that the lab rooms aren't overbooked.

**LLM**

> Lab capacity check by time slot:
>
> Monday:
> - Slot 3: Physics (Science Lab)
> - Slot 4: Networks (CS), Chemistry (Science) ✓
>
> Wednesday:  
> - Slot 1: AI (CS)
> - Slot 2: Database (CS), Physics (Science)
> - Slot 3: Chemistry (Science) 
> - Slot 4: ML (CS), Security (CS), Biology (Science) ✓
>
> Friday:
> - Slot 1: Networks (CS) 
> - Slot 2: AI (CS)
> - Slot 3: Database (CS)
> - Slot 4: ML (CS), Security (CS), Biology (Science) ✓
>
> Each slot has ≤2 CS courses and ≤1 science course, so the schedule is valid for labs.

---

MCP server version 0.2.1
Claude 3.5 Sonnet