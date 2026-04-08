(define (problem parole_office)
    (:domain office)

    ; Begin objects
    (:objects
        robot - agent
        bathroom bedroom corridor_1 corridor_2 dining_room kitchen living_room - room
        locker shelf paper book desk lamp - item
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
        (item_at desk living_room)
        (item_at locker bathroom)
        (item_in paper locker)
        (item_at shelf bedroom)
        (item_in book shelf)
        (item_at lamp bedroom)

        ; Attributes
        (item_accessible desk)
        (item_pickable desk)
        (item_loadable locker)
        (item_pickable paper)
        (item_loadable shelf)
        (item_pickable book)
        (item_pickable lamp)
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (item_at desk living_room)
            (item_at locker living_room)
            (item_at shelf living_room)
            (item_in paper locker)
            (item_in book shelf)
            (item_at lamp living_room)
        )
    )
    ; End goal

)