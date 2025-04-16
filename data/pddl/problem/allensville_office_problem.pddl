(define (problem allensville_office)
    (:domain office)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 bedroom_1 bedroom_2 corridor_1 corridor_2 corridor_3 dining_room kitchen living_room lobby - room
        locker shelf paper book desk lamp - item
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
        (item_at desk living_room)
        (item_at locker lobby)
        (item_in paper locker)
        (item_at shelf bedroom_1)
        (item_in book shelf)
        (item_at lamp bedroom_2)

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