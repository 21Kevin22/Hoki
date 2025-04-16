(define (problem parole)
    (:domain human)

    ; Begin objects
    (:objects
        robot - agent
        tom - human
        bathroom bedroom corridor_1 corridor_2 dining_room kitchen living_room - room
        mug coffee_machine - item
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
        (human_at tom bedroom)
        (item_at mug dining_room)
        (item_at coffee_machine kitchen)

        ; Attributes
        (item_is_mug mug)
        (item_is_coffee_machine coffee_machine)
        (item_pickable mug)
        (item_accessible mug)
        (item_turnable coffee_machine)
        (item_accessible coffee_machine)
        (not(mug_filled mug))
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (mug_filled mug)
            (human_has_item tom mug)
        )
    )
    ; End goal
)