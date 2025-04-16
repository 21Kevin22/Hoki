;Header and description
(define (domain office)

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item
    )
    ; End types

    ; Begin predicates
    (:predicates
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (item_in ?i2 - item ?i1 - item)
        (item_pickable ?i - item)
        (item_loadable ?i - item)
        (item_accessible ?i - item)
        (item_empty ?i - item)

        (neighbor ?r1 - room ?r2 - room)

        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
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

    (:action pick_loadable
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_loadable ?i)
            (item_empty ?i)
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

    (:action drop_loadable
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (item_loadable ?i)
            (item_empty ?i)
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

    (:action load
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_loadable ?i1)
            (item_pickable ?i2)
            (not(item_in ?i2 ?i1))
            (agent_loaded ?a)
            (agent_has_item ?a ?i2)
            (item_empty ?i1)
        )
        :effect (and
            (not(item_at ?i2 ?r))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i2))
            (item_in ?i2 ?i1)
            (not(item_empty ?i1))
        )
    )

    (:action unload
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (not(item_at ?i2 ?r))
            (item_loadable ?i1)
            (item_pickable ?i2)
            (item_in ?i2 ?i1)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i2))
            (not(item_empty ?i1))
        )
        :effect (and
            (item_at ?i2 ?r)
            (not(item_in ?i2 ?i1))
            (item_empty ?i1)
        )
    )
    ; End actions
)