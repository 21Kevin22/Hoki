(define (problem allensville_pc)
    (:domain pc)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 bedroom_1 bedroom_2 corridor_1 corridor_2 corridor_3 dining_room kitchen living_room lobby - room
        mainboard cpu ram ssd gpu psu - item
        my_pc - pc
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
        (item_at mainboard bedroom_1)
        (item_at cpu bedroom_2)
        (item_at ram lobby)
        (item_at ssd dining_room)
        (item_at gpu bathroom_2)
        (item_at psu bathroom_1)

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