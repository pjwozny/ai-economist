# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicPlanner(BaseAgent):
    """
    A basic planner agent represents a social planner that sets macroeconomic policy.

    Unlike the "mobile" agent, the planner does not represent an embodied agent in
    the world environment. BasicPlanner modifies the BaseAgent class to remove
    location as part of the agent state.

    Also unlike the "mobile" agent, the planner agent is expected to be unique --
    that is, there should only be 1 planner. For this reason, BasicPlanner ignores
    the idx argument during construction and always sets its agent index as "p".
    """

    name = "BasicPlanner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.state["loc"]

        # Overwrite any specified index so that this one is always indexed as 'p'
        # (make a separate class of planner if you want there to be multiple planners
        # in a game)
        self._idx = "p"

    @property
    def loc(self):
        """
        Planner agents do not occupy any location.
        """
        raise AttributeError("BasicPlanner agents do not occupy a location.")

    @agent_registry.add
    class MultiPlanner(BaseAgent):
            """
            A multi planner agent represents a social planner that sets macroeconomic policy.
            A multiplanner is identical to a basic planner except that there is one multi-planner per state.

            Unlike the "mobile" agent, the planner does not represent an embodied agent in
            the world environment. BasicPlanner modifies the BaseAgent class to remove
            location as part of the agent state.

            Multiplanner idxs are p + a number
            """
            name = "MultiPlanner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.state["loc"]
        self.country = int(self.id[2:])

    @property
    def loc(self):
        """
        Planner agents do not occupy any location.
        """
        raise AttributeError("BasicPlanner agents do not occupy a location.")

    def register_components(self, components):
        """Used during environment construction to set up state/action spaces."""
        assert not self._registered_components
        for component in components:
            if component.name == 'PeriodicTaxBracketMultiOrchestrator':
                n = component.get_n_actions(self.name, self.country)
            else:
                n = component.get_n_actions(self.name)
            if n is None:
                continue

            # Most components will have a single action-per-agent, so n is an int
            if isinstance(n, int):
                if n == 0:
                    continue
                self._incorporate_component(component.name, n)

            # They can also internally handle multiple actions-per-agent,
            # so n is an tuple or list
            elif isinstance(n, (tuple, list)):
                for action_sub_name, n_ in n:
                    if n_ == 0:
                        continue
                    if "." in action_sub_name:
                        raise NameError(
                            "Sub-action {} of component {} "
                            "is illegally named.".format(
                                action_sub_name, component.name
                            )
                        )
                    self._incorporate_component(
                        "{}.{}".format(component.name, action_sub_name), n_
                    )

            # If that's not what we got something is funky.
            else:
                raise TypeError(
                    "Received unexpected type ({}) from {}.get_n_actions('{}')".format(
                        type(n), component.name, self.name
                    )
                )

            for k, v in component.get_additional_state_fields(self.name).items():
                self.state[k] = v

        # Currently no actions are available to this agent. Give it a placeholder.
        if len(self.action) == 0 and self.multi_action_mode:
            self._incorporate_component("PassiveAgentPlaceholder", 0)
            self._passive_multi_action_agent = True

        elif len(self.action) == 1 and not self.multi_action_mode:
            self._one_component_single_action = True
            self._premask = np.ones(1 + self._total_actions, dtype=np.float32)

        self._registered_components = True

        self._noop_action_dict = {k: v * 0 for k, v in self.action.items()}

        verbose = False
        if verbose:
            print(self.name, self.idx, "constructed action map:")
            for k, v in self.single_action_map.items():
                print("single action map:", k, v)
            for k, v in self.action.items():
                print("action:", k, v)
            for k, v in self.action_dim.items():
                print("action_dim:", k, v)

    
