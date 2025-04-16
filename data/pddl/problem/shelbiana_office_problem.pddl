(define (problem shelbiana_office)
    (:domain office)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 balcony bedroom_1 bedroom_2 bedroom_3 closet corridor_1 corridor_2 kitchen living_room lobby - room
        locker shelf paper book desk lamp - item
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
        (item_at desk living_room)
        (item_at locker closet)
        (item_in paper locker)
        (item_at shelf lobby)
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