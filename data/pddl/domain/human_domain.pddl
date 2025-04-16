;Header and description
(define (domain human)

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item human
    )
    ; End types

    ; Begin predicates
    (:predicates
        (agent_at ?a - agent ?r - room)
        (human_at ?h - human ?r - room)
        (item_at ?i - item ?r - room)
        (item_pickable ?i - item)
        (item_turnable ?i - item)
        (item_accessible ?i - item)

        (neighbor ?r1 - room ?r2 - room)

        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)

        (human_has_item ?h - human ?i - item)

        (item_is_mug ?i - item)
        (item_is_coffee_machine ?i - item)
        (mug_filled ?i - item)
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

    (:action pass
        :parameters (?a - agent ?i - item ?h - human ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (human_at ?h ?r)
            (item_accessible ?i)
            (item_pickable ?i)
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
            (not(human_has_item ?h ?i))
        )
        :effect (and
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
            (human_has_item ?h ?i)
        )
    )

    (:action make_coffee
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_at ?i2 ?r)
            (item_accessible ?i1)
            (item_accessible ?i2)
            (item_is_mug ?i1)
            (item_is_coffee_machine ?i2)
            (item_pickable ?i1)
            (item_turnable ?i2)
            (not(mug_filled ?i1))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i1))
        )
        :effect (and
            (mug_filled ?i1)
        )
    )
    ; End actions

)