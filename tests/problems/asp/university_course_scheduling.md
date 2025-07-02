# University Course Scheduling

## Problem Description

A university department needs to schedule a set of courses for the upcoming semester. Each course must be assigned a time slot and a classroom. The problem must satisfy the following constraints:

No Overlapping Courses for Instructors: An instructor cannot teach more than one course at the same time.

Room Capacity: Each classroom has a limited capacity, and the assigned classroom must be able to accommodate all enrolled students.

Course Conflicts: Some courses cannot be scheduled at the same time because students are likely to enroll in both.

Limited Time Slots: There are a fixed number of time slots available each day.

Room Availability: Some rooms are not available at certain times.

## Details:

5 courses: CS101, CS102, CS201, CS202, CS301

3 instructors: Dr. Smith, Dr. Lee, Dr. Patel

3 classrooms: Room A (30 seats), Room B (50 seats), Room C (20 seats)

4 time slots: Mon 9am, Mon 11am, Wed 9am, Wed 11am

Enrollment numbers: e.g., CS101 (40 students), CS201 (25 students)

Room C is unavailable on Mon 9am

Conflict: CS101 and CS201 cannot be scheduled at the same time
