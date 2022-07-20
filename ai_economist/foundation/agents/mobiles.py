# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from unicodedata import name
from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicMobileAgent(BaseAgent):
    """
    A basic mobile agent represents an individual actor in the economic simulation.

    "Mobile" refers to agents of this type being able to move around in the 2D world.
    """

    name = "BasicMobileAgent"

@agent_registry.add
class Citizen(BaseAgent):
  name = "Citizen"
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.state["nation"] = 0

  @property
  def nation(self):
      """Returns nation index

      Example:
          >> self.nation
          0
      """
      return self.state["nation"]
