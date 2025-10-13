"""
DataGenius PRO - Base Agent Class
Abstract base class for all AI agents in the system
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Standard result format for all agents"""
    
    agent_name: str
    status: str = Field(default="success", description="success, failed, partial")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    def is_success(self) -> bool:
        """Check if agent execution was successful"""
        return self.status == "success"
    
    def add_error(self, error: str) -> None:
        """Add error message"""
        self.errors.append(error)
        if self.status == "success":
            self.status = "failed"
    
    def add_warning(self, warning: str) -> None:
        """Add warning message"""
        self.warnings.append(warning)
        if self.status == "success":
            self.status = "partial"


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents
    
    All agents should inherit from this class and implement:
    - execute() method
    - validate_input() method (optional)
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logger.bind(agent=name)
        self._result: Optional[AgentResult] = None
    
    @abstractmethod
    def execute(self, **kwargs) -> AgentResult:
        """
        Main execution method - must be implemented by subclasses
        
        Args:
            **kwargs: Agent-specific parameters
        
        Returns:
            AgentResult with execution results
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters (optional override)
        
        Args:
            **kwargs: Parameters to validate
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If validation fails
        """
        return True
    
    def before_execute(self, **kwargs) -> None:
        """
        Hook called before execute() (optional override)
        """
        self.logger.info(f"[{self.name}] Starting execution")
    
    def after_execute(self, result: AgentResult) -> None:
        """
        Hook called after execute() (optional override)
        """
        self.logger.info(
            f"[{self.name}] Execution completed "
            f"(status: {result.status}, time: {result.execution_time:.2f}s)"
        )
    
    def run(self, **kwargs) -> AgentResult:
        """
        Main entry point - handles execution lifecycle
        
        Args:
            **kwargs: Agent-specific parameters
        
        Returns:
            AgentResult with execution results
        """
        import time
        
        try:
            # Validate input
            self.validate_input(**kwargs)
            
            # Before hook
            self.before_execute(**kwargs)
            
            # Execute
            start_time = time.time()
            result = self.execute(**kwargs)
            result.execution_time = time.time() - start_time
            
            # Store result
            self._result = result
            
            # After hook
            self.after_execute(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Execution failed: {e}", exc_info=True)
            
            # Create error result
            result = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
            )
            result.add_error(str(e))
            
            self._result = result
            return result
    
    def get_last_result(self) -> Optional[AgentResult]:
        """Get result from last execution"""
        return self._result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class PipelineAgent(BaseAgent):
    """
    Agent that orchestrates multiple sub-agents in a pipeline
    """
    
    def __init__(self, name: str, agents: List[BaseAgent], description: str = ""):
        super().__init__(name, description)
        self.agents = agents
    
    def execute(self, **kwargs) -> AgentResult:
        """Execute all agents in sequence"""
        
        result = AgentResult(agent_name=self.name)
        pipeline_results = []
        
        for agent in self.agents:
            self.logger.info(f"[Pipeline: {self.name}] Executing {agent.name}")
            
            # Run agent
            agent_result = agent.run(**kwargs)
            pipeline_results.append(agent_result)
            
            # Check if failed
            if not agent_result.is_success():
                result.status = "failed"
                result.add_error(
                    f"Agent {agent.name} failed: {', '.join(agent_result.errors)}"
                )
                break
            
            # Pass data to next agent
            kwargs.update(agent_result.data)
        
        # Aggregate results
        result.data = {
            "pipeline_results": pipeline_results,
            "agents_executed": len(pipeline_results),
            "final_data": kwargs,
        }
        
        return result
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add agent to pipeline"""
        self.agents.append(agent)
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove agent from pipeline by name"""
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                self.agents.pop(i)
                return True
        return False


class ParallelAgent(BaseAgent):
    """
    Agent that executes multiple sub-agents in parallel
    """
    
    def __init__(self, name: str, agents: List[BaseAgent], description: str = ""):
        super().__init__(name, description)
        self.agents = agents
    
    def execute(self, **kwargs) -> AgentResult:
        """Execute all agents in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        result = AgentResult(agent_name=self.name)
        parallel_results = []
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all agents
            futures = {
                executor.submit(agent.run, **kwargs): agent
                for agent in self.agents
            }
            
            # Collect results
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    agent_result = future.result()
                    parallel_results.append(agent_result)
                    
                    # Check for failures
                    if not agent_result.is_success():
                        result.add_warning(
                            f"Agent {agent.name} failed: {', '.join(agent_result.errors)}"
                        )
                
                except Exception as e:
                    result.add_error(f"Agent {agent.name} raised exception: {e}")
        
        # Aggregate results
        result.data = {
            "parallel_results": parallel_results,
            "agents_executed": len(parallel_results),
        }
        
        # Set status based on results
        if all(r.is_success() for r in parallel_results):
            result.status = "success"
        elif any(r.is_success() for r in parallel_results):
            result.status = "partial"
        else:
            result.status = "failed"
        
        return result