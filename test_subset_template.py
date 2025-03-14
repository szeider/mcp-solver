#!/usr/bin/env python

"""
Test script for the smallest_subset_with_property template.
This script tests finding the smallest subset of tasks that cannot be scheduled together.
"""

from z3 import *
from mcp_solver.z3.templates import smallest_subset_with_property

def test_task_scheduling():
    """Test the task scheduling example from the documentation."""
    print("Testing smallest_subset_with_property with task scheduling example...")
    
    # Task data: (task_id, start_time, end_time)
    tasks = [
        ('A', 9, 10),   # Task A: 9 AM to 10 AM
        ('B', 9, 11),   # Task B: 9 AM to 11 AM
        ('C', 10, 12),  # Task C: 10 AM to 12 PM
        ('D', 11, 13),  # Task D: 11 AM to 1 PM
        ('E', 12, 14)   # Task E: 12 PM to 2 PM
    ]
    
    # Property checker: returns True if tasks CANNOT be scheduled without conflicts
    def is_unschedulable(task_subset):
        if len(task_subset) <= 1:
            return False  # A single task is always schedulable
            
        s = Solver()
        
        # Create boolean variables for whether each task is doable
        can_do = {}
        for task_id, start, end in task_subset:
            can_do[task_id] = Bool(f"can_do_{task_id}")
            s.add(can_do[task_id])  # We want to do all tasks
        
        # Add constraints for conflicting tasks
        for i, (task1_id, start1, end1) in enumerate(task_subset):
            for task2_id, start2, end2 in task_subset[i+1:]:
                # Tasks conflict if their time ranges overlap
                if not (end1 <= start2 or end2 <= start1):
                    # If tasks overlap, we can't do both
                    s.add(Not(And(can_do[task1_id], can_do[task2_id])))
        
        # If unsatisfiable, the tasks cannot all be scheduled
        return s.check() == unsat
    
    # Optional: Provide candidate subsets to speed up search
    is_unschedulable.candidate_subsets = [
        [tasks[0], tasks[1]],  # A and B
        [tasks[1], tasks[2]]   # B and C
    ]
    
    # Find the smallest unschedulable subset
    smallest = smallest_subset_with_property(tasks, is_unschedulable, min_size=2)
    
    if smallest:
        print("\nSmallest unschedulable subset of tasks:")
        for task in smallest:
            print(f"Task {task[0]}: {task[1]} to {task[2]}")
            
        # Verify the result
        print("\nVerifying that this subset is unschedulable...")
        assert is_unschedulable(smallest), "Error: The found subset should be unschedulable!"
        
        # Verify that it's minimal
        for i in range(len(smallest)):
            smaller_subset = smallest[:i] + smallest[i+1:]
            if len(smaller_subset) > 0:
                is_smaller_unschedulable = is_unschedulable(smaller_subset)
                print(f"Subset without task {smallest[i][0]} is unschedulable: {is_smaller_unschedulable}")
                assert not is_smaller_unschedulable, f"Error: Found a smaller unschedulable subset!"
                
        print("\nTests passed! The subset is minimal and unschedulable.")
    else:
        print("No unschedulable subset found (this shouldn't happen with our example).")
        assert False, "Failed to find an unschedulable subset!"

def test_server_resilience():
    """Test the server resilience example from the documentation."""
    print("\nTesting smallest_subset_with_property with server resilience example...")
    
    # Network data: (server_id, capacity, services)
    servers = [
        ('S1', 50, ['web', 'auth']),
        ('S2', 75, ['web', 'database']),
        ('S3', 60, ['auth', 'cache']),
        ('S4', 80, ['web', 'api']),
        ('S5', 65, ['database', 'cache']),
        ('S6', 70, ['api', 'storage'])
    ]
    
    # Required services and minimum capacity
    required_services = ['web', 'auth', 'database', 'api']
    MIN_TOTAL_CAPACITY = 150
    
    # Property checker: returns True if taking these servers offline would break the network
    def breaks_network(offline_servers):
        """Check if removing these servers would make the network non-functional"""
        if not offline_servers:
            return False  # No servers offline can't break the network
        
        # Find which servers remain online
        online_servers = [s for s in servers if s not in offline_servers]
        
        # Check if minimum capacity is maintained
        total_capacity = sum(capacity for _, capacity, _ in online_servers)
        if total_capacity < MIN_TOTAL_CAPACITY:
            return True  # Network broken due to insufficient capacity
        
        # Check if all required services are still available
        available_services = set()
        for _, _, services in online_servers:
            available_services.update(services)
        
        # If any required service is missing, the network is broken
        for service in required_services:
            if service not in available_services:
                return True
        
        # Network still functional
        return False
    
    # Find the smallest subset of servers that would break the network
    smallest = smallest_subset_with_property(servers, breaks_network, min_size=1)
    
    if smallest:
        print("\nSmallest critical set of servers:")
        for server in smallest:
            print(f"Server {server[0]}: Capacity {server[1]}, Services {server[2]}")
        
        # Verify the result
        print("\nVerifying that this subset breaks the network...")
        assert breaks_network(smallest), "Error: The found subset should break the network!"
        
        # Verify that it's minimal
        for i in range(len(smallest)):
            smaller_subset = smallest[:i] + smallest[i+1:]
            if len(smaller_subset) > 0:
                is_smaller_critical = breaks_network(smaller_subset)
                print(f"Subset without server {smallest[i][0]} breaks network: {is_smaller_critical}")
                assert not is_smaller_critical, f"Error: Found a smaller critical subset!"
                
        print("\nTests passed! The subset is minimal and critical.")
    else:
        print("No critical subset found (this shouldn't happen with our example).")
        assert False, "Failed to find a critical subset!"

if __name__ == "__main__":
    test_task_scheduling()
    test_server_resilience()
    print("\nAll tests completed successfully!") 