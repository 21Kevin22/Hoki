(define (problem shelbiana_pc)
    (:domain pc)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 balcony bedroom_1 bedroom_2 bedroom_3 closet corridor_1 corridor_2 kitchen living_room lobby - room
        mainboard cpu ram ssd gpu psu - item
        my_pc - pc
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
        (item_at psu bathroom_2)
        (item_at cpu bedroom_1)
        (item_at ssd bedroom_2)
        (item_at gpu closet)
        (item_at ram kitchen)
        (item_at mainboard bathroom_1)

        ; Attributes
        (item_is_mainboard mainboard)
        (item_accessible mainboard)
        (item_pickable mainboard)
        (item_is_cpu cpu)
        (item_accessible cpu)
        (item_pickable cpu)
        (item_is_ram ram)
        (item_accessible ram)
        (item_pickable ram)
        (item_is_ssd ssd)
        (item_accessible ssd)
        (item_pickable ssd)
        (item_is_gpu gpu)
        (item_accessible gpu)
        (item_pickable gpu)
        (item_is_psu psu)
        (item_accessible psu)
        (item_pickable psu)
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (item_at psu living_room)
            (item_at gpu living_room)
            (item_at ssd living_room)
            (item_at ram living_room)
            (item_at cpu living_room)
            (item_at mainboard living_room)
            (pc_assembled my_pc)
        )
    )
    ; End goal
)