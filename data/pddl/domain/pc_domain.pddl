;Header and description
(define (domain pc)

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item pc
    )
    ; End types

    ; Begin predicates
    (:predicates
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)

        (item_pickable ?i - item)
        (item_accessible ?i - item)

        (neighbor ?r1 - room ?r2 - room)

        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)

        (item_is_mainboard ?i - item)
        (item_is_cpu ?i - item)
        (item_is_ram ?i - item)
        (item_is_ssd ?i - item)
        (item_is_gpu ?i - item)
        (item_is_psu ?i - item)

        (pc_assembled ?b - pc)
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

    (:action assemble
        :parameters (?a - agent ?r - room ?i1 - item ?i2 - item ?i3 - item ?i4 - item ?i5 - item ?i6 - item ?p - pc)
        :precondition (and
            (not(pc_assembled ?p))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i1))
            (not(agent_has_item ?a ?i2))
            (not(agent_has_item ?a ?i3))
            (not(agent_has_item ?a ?i4))
            (not(agent_has_item ?a ?i5))
            (not(agent_has_item ?a ?i6))
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_at ?i2 ?r)
            (item_at ?i3 ?r)
            (item_at ?i4 ?r)
            (item_at ?i5 ?r)
            (item_at ?i6 ?r)
            (item_is_mainboard ?i1)
            (item_is_cpu ?i2)
            (item_is_ram ?i3)
            (item_is_ssd ?i4)
            (item_is_gpu ?i5)
            (item_is_psu ?i6)
            (item_accessible ?i1)
            (item_accessible ?i2)
            (item_accessible ?i3)
            (item_accessible ?i4)
            (item_accessible ?i5)
            (item_accessible ?i6)
            (item_pickable ?i1)
            (item_pickable ?i2)
            (item_pickable ?i3)
            (item_pickable ?i4)
            (item_pickable ?i5)
            (item_pickable ?i6)
        )
        :effect (and
            (pc_assembled ?p)
        )
    )
    ; End actions

)