(define (problem parole_dining)
    (:domain dining)

    ; Begin objects
    (:objects
        robot - agent
        bathroom bedroom corridor_1 corridor_2 dining_room kitchen living_room - room
        plate fork knife spoon glass flower dining_table - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom corridor_2)
        (neighbor bedroom corridor_2)
        (neighbor corridor_1 corridor_2)
        (neighbor corridor_1 dining_room)
        (neighbor corridor_1 kitchen)
        (neighbor corridor_2 bathroom)
        (neighbor corridor_2 bedroom)
        (neighbor corridor_2 corridor_1)
        (neighbor dining_room corridor_1)
        (neighbor dining_room living_room)
        (neighbor kitchen corridor_1)
        (neighbor living_room dining_room)

        ; Positions
        (agent_at robot living_room)
        (item_at plate living_room)
        (item_at fork kitchen)
        (item_at knife kitchen)
        (item_at spoon kitchen)
        (item_at glass bedroom)
        (item_at flower corridor_1)
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