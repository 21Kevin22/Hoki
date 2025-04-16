(define (problem allensville_dining)
    (:domain dining)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 bedroom_1 bedroom_2 corridor_1 corridor_2 corridor_3 dining_room kitchen living_room lobby - room
        plate fork knife spoon glass flower dining_table - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom_1 corridor_2)
        (neighbor bathroom_2 corridor_3)
        (neighbor bedroom_1 corridor_2)
        (neighbor bedroom_2 corridor_3)
        (neighbor corridor_1 lobby)
        (neighbor corridor_1 corridor_3)
        (neighbor corridor_2 bathroom_1)
        (neighbor corridor_2 bedroom_1)
        (neighbor corridor_2 corridor_3)
        (neighbor corridor_3 corridor_1)
        (neighbor corridor_3 corridor_2)
        (neighbor corridor_3 bathroom_2)
        (neighbor corridor_3 bedroom_2)
        (neighbor corridor_3 kitchen)
        (neighbor corridor_3 living_room)
        (neighbor dining_room kitchen)
        (neighbor dining_room living_room)
        (neighbor kitchen corridor_3)
        (neighbor kitchen dining_room)
        (neighbor living_room corridor_3)
        (neighbor living_room dining_room)
        (neighbor lobby corridor_1)

        ; Positions
        (agent_at robot living_room)
        (item_at plate bedroom_2)
        (item_at fork kitchen)
        (item_at knife kitchen)
        (item_at spoon kitchen)
        (item_at glass bedroom_1)
        (item_at flower lobby)
        (item_at dining_table dining_room)

        ; Attributes
        (item_is_dining_table dining_table)
        (item_pickable plate)
        (item_accessible plate)
        (item_pickable fork)
        (item_accessible fork)
        (item_pickable knife)
        (item_accessible knife)
        (item_pickable spoon)
        (item_accessible spoon)
        (item_pickable glass)
        (item_accessible glass)
        (item_pickable flower)
        (item_accessible flower)
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (item_on plate dining_table)
            (item_on fork dining_table)
            (item_on knife dining_table)
            (item_on spoon dining_table)
            (item_on glass dining_table)
            (item_on flower dining_table)
        )
    )
    ; End goal

)