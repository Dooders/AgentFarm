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
    
    def update_position(self, new_position, spatial_service=None):
        """Update resource position and optionally register as mobile.
        
        Args:
            new_position (tuple): New (x, y) position coordinates
            spatial_service: Optional spatial service to register mobility with
        """
        if self.position != new_position:
            self.position = new_position
            # Register as mobile resource for selective tracking optimization
            if spatial_service and hasattr(spatial_service, 'register_mobile_resource'):
                try:
                    spatial_service.register_mobile_resource(str(self.resource_id))
                    spatial_service.mark_positions_dirty()
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to register resource {self.resource_id} as mobile: {e}")
    
    def set_mobile(self, mobile: bool = True, spatial_service=None) -> None:
        """Set whether this resource should be tracked as mobile for spatial indexing optimization.
        
        Mobile resources are checked for position changes on every spatial index update.
        Static (immobile) resources are hashed once and cached for better performance.
        
        Args:
            mobile (bool): True to track as mobile, False to treat as static
            spatial_service: Spatial service to register/unregister with
        """
        if not spatial_service:
            return
            
        try:
            resource_id = str(self.resource_id)
            if mobile:
                if hasattr(spatial_service, 'register_mobile_resource'):
                    spatial_service.register_mobile_resource(resource_id)
            else:
                if hasattr(spatial_service, 'unregister_mobile_resource'):
                    spatial_service.unregister_mobile_resource(resource_id)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to set mobility status for resource {self.resource_id}: {e}")
