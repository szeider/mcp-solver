from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any


class SolverManager(ABC):
    """
    Abstract base class for solver managers.
    This class defines the interface that all solver implementations must follow.
    """

    def __init__(self):
        """
        Initialize a new solver manager.
        """
        self.initialized = False
        self.last_solve_time = None

    @abstractmethod
    async def clear_model(self) -> dict[str, Any]:
        """
        Clear the current model.

        Returns:
            A dictionary with a message indicating the model was cleared
        """
        pass

    @abstractmethod
    async def add_item(self, index: int, content: str) -> dict[str, Any]:
        """
        Add an item to the model at the specified index.

        Args:
            index: The index at which to add the item
            content: The content to add

        Returns:
            A dictionary with the result of the operation
        """
        pass

    @abstractmethod
    async def delete_item(self, index: int) -> dict[str, Any]:
        """
        Delete an item from the model at the specified index.

        Args:
            index: The index of the item to delete

        Returns:
            A dictionary with the result of the operation
        """
        pass

    @abstractmethod
    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """
        Replace an item in the model at the specified index.

        Args:
            index: The index of the item to replace
            content: The new content

        Returns:
            A dictionary with the result of the operation
        """
        pass

    @abstractmethod
    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
        """
        Solve the current model.

        Args:
            timeout: Timeout for the solve operation

        Returns:
            A dictionary with the result of the solve operation
        """
        pass

    @abstractmethod
    def get_solution(self) -> dict[str, Any]:
        """
        Get the current solution.

        Returns:
            A dictionary with the current solution
        """
        pass

    @abstractmethod
    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
        """
        Get the value of a variable from the current solution.

        Args:
            variable_name: The name of the variable

        Returns:
            A dictionary with the value of the variable
        """
        pass

    @abstractmethod
    def get_solve_time(self) -> dict[str, Any]:
        """
        Get the time taken for the last solve operation.

        Returns:
            A dictionary with the solve time information
        """
        pass
