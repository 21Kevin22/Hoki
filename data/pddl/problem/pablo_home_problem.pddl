(define (problem pablo)
    (:domain home)

    ; Begin objects
    (:objects
        robot - agent
        bathroom bedroom closet corridor living_room - room
        sink bottle toilet book vase chair bed couch tv - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom corridor)
        (neighbor bedroom living_room)
        (neighbor closet corridor)
        (neighbor corridor bathroom)
        (neighbor corridor closet)
        (neighbor corridor living_room)
        (neighbor living_room bedroom)
        (neighbor living_room corridor)

        ; Positions
        (agent_at robot living_room)

        (item_at sink bathroom)
        (item_at bottle bathroom)
        (item_at toilet bathroom)

        (item_at book bedroom)
        (item_at vase bedroom)
        (item_at chair bedroom)
        (item_at bed bedroom)

        (item_at couch living_room)
        (item_at tv living_room)

        ; Attributes
        (item_pickable bottle)
        (item_accessible bottle)
        (item_pickable book)
        (item_accessible book)
        (item_pickable vase)
        (item_accessible vase)
        (item_turnable tv)
        (item_accessible tv)
    )
    ; End init

    ; Begin goal
    (:goal
        (item_at bottle living_room)
    )
    ; End goal
)