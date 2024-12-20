class Resource:
    """Resource node in the environment that agents can gather from.

    Attributes:
        resource_id (int): Unique identifier for the resource
        position (tuple): (x,y) coordinates in the environment
        amount (float): Current amount of resource available
        max_amount (float): Maximum amount this resource can hold
        regeneration_rate (float): Rate at which resource regenerates
    """

    def __init__(
        self, resource_id, position, amount, max_amount=None, regeneration_rate=0.1
    ):
        self.resource_id = resource_id
        self.position = position  # (x, y) coordinates
        self.amount = amount
        self.max_amount = max_amount if max_amount is not None else amount
        self.regeneration_rate = regeneration_rate

    def is_depleted(self):
        """Check if resource is depleted."""
        return self.amount <= 0

    def consume(self, consumption_amount):
        """Consume some amount of the resource."""
        self.amount -= consumption_amount
        if self.amount < 0:
            self.amount = 0

    def regenerate(self, regen_amount):
        """Regenerate some amount of the resource up to max_amount."""
        if self.max_amount is not None:
            self.amount = min(self.amount + regen_amount, self.max_amount)
        else:
            self.amount += regen_amount
