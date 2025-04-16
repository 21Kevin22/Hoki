(define (problem kemblesville)
    (:domain laundry)

    ; Begin objects
    (:objects
        robot - agent
        bathroom closet_1 closet_2 corridor_1 corridor_2 bedroom_1 bedroom_2 living_room kitchen - room
        detergent clothes wash_machine - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom corridor_1)
        (neighbor bathroom closet_1)
        (neighbor closet_1 bathroom)
        (neighbor closet_1 bedroom_1)
        (neighbor closet_2 corridor_2)
        (neighbor closet_2 kitchen)
        (neighbor corridor_1 bedroom_1)
        (neighbor corridor_1 bedroom_2)
        (neighbor corridor_1 bathroom)
        (neighbor corridor_1 living_room)
        (neighbor corridor_2 closet_2)
        (neighbor corridor_2 kitchen)
        (neighbor corridor_2 living_room)
        (neighbor bedroom_1 corridor_1)
        (neighbor bedroom_1 closet_1)
        (neighbor bedroom_2 corridor_1)
        (neighbor living_room corridor_1)
        (neighbor living_room corridor_2)
        (neighbor kitchen corridor_2)
        (neighbor kitchen closet_2)

        ; Positions
        (agent_at robot living_room)
        (item_at detergent bathroom)
        (item_at clothes closet_1)
        (item_at wash_machine kitchen)

        ; Attributes
        (item_is_cloth clothes)
        (item_is_detergent detergent)
        (item_is_wash_machine wash_machine)
        (item_pickable clothes)
        (item_accessible clothes)
        (item_turnable wash_machine)
        (item_accessible wash_machine)
        (item_pickable detergent)
        (item_accessible detergent)
        (not(cloth_clean clothes))
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (cloth_clean clothes)
            (item_at clothes bedroom_1)
        )
    )
    ; End goal
)