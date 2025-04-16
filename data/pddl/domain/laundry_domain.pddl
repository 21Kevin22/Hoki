;Header and description
(define (domain laundry)

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item
    )
    ; End types

    ; Begin predicates
    (:predicates
        (neighbor ?r1 - room ?r2 - room)
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (item_pickable ?i - item)
        (item_turnable ?i - item)
        (item_accessible ?i - item)
        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
        (item_is_cloth ?i - item)
        (item_is_detergent ?i - item)
        (item_is_wash_machine ?i - item)
        (cloth_clean ?i - item)
    )
    ; End predicates

    ; Begin actions
    (:action goto
        :parameters (?a - agent ?r1 - room ?r2 - room)
        :precondition (and
            (agent_at ?a ?r1)
            (neighbor ?r1 ?r2)
        )
        :effect (and
            (not(agent_at ?a ?r1))
            (agent_at ?a ?r2)
        )
    )

    (:action pick
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_accessible ?i)
            (item_pickable ?i)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
        )
        :effect (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
        )
    )

    (:action drop
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (item_accessible ?i)
            (item_pickable ?i)
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
        )
        :effect (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
        )
    )

    (:action launder
        :parameters (?a - agent ?i1 - item ?i2 - item ?i3 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_at ?i2 ?r)
            (item_at ?i3 ?r)
            (item_accessible ?i1)
            (item_accessible ?i2)
            (item_accessible ?i3)
            (item_is_cloth ?i1)
            (item_is_detergent ?i2)
            (item_is_wash_machine ?i3)
            (item_pickable ?i1)
            (item_pickable ?i2)
            (item_turnable ?i3)
            (not(cloth_clean ?i1))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i1))
        )
        :effect (and
            (cloth_clean ?i1)
        )
    )
    ; End actions

)