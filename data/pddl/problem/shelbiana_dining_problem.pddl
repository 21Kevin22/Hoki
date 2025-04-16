(define (problem shelbiana_dining)
    (:domain dining)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 balcony bedroom_1 bedroom_2 bedroom_3 closet corridor_1 corridor_2 kitchen living_room lobby - room
        plate fork knife spoon glass flower dining_table - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom_1 lobby)
        (neighbor bathroom_2 corridor_2)
        (neighbor balcony living_room)
        (neighbor balcony bedroom_3)
        (neighbor closet corridor_2)
        (neighbor corridor_1 corridor_2)
        (neighbor corridor_1 kitchen)
        (neighbor corridor_1 living_room)
        (neighbor corridor_2 bathroom_2)
        (neighbor corridor_2 bedroom_2)
        (neighbor corridor_2 bedroom_3)
        (neighbor corridor_2 closet)
        (neighbor corridor_2 corridor_1)
        (neighbor living_room balcony)
        (neighbor living_room corridor_1)
        (neighbor living_room lobby)
        (neighbor bedroom_1 lobby)
        (neighbor bedroom_2 corridor_2)
        (neighbor bedroom_3 balcony)
        (neighbor bedroom_3 corridor_2)
        (neighbor lobby bathroom_1)
        (neighbor lobby bedroom_1)
        (neighbor lobby living_room)
        (neighbor kitchen corridor_1)

        ; Positions
        (agent_at robot bedroom_3)
        (item_at plate living_room)
        (item_at fork kitchen)
        (item_at knife kitchen)
        (item_at spoon kitchen)
        (item_at glass corridor_2)
        (item_at flower lobby)
        (item_at dining_table balcony)

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