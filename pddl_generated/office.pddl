(define (domain office)
  (:requirements :strips :typing :negative-preconditions)
  (:types agent room item)
  (:predicates
    (neighbor ?r1 - room ?r2 - room)
    (agent_at ?a - agent ?r - room)
    (item_at ?i - item ?r - room)
    (agent_loaded ?a - agent)
    (agent_has_item ?a - agent ?i - item)
    (can_graspable ?i - item)
    (can_liftable ?i - item)
    (can_accessible ?i - item)
  )

  (:action goto
    :parameters (?a - agent ?r1 - room ?r2 - room)
    :precondition (and (agent_at ?a ?r1) (neighbor ?r1 ?r2))
    :effect (and (not (agent_at ?a ?r1)) (agent_at ?a ?r2))
  )

  (:action pick
    :parameters (?a - agent ?i - item ?r - room)
    :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (can_accessible ?i) (can_graspable ?i) (can_liftable ?i) (not (agent_loaded ?a)))
    :effect (and (not (item_at ?i ?r)) (agent_loaded ?a) (agent_has_item ?a ?i))
  )

  (:action drop
    :parameters (?a - agent ?i - item ?r - room)
    :precondition (and (agent_at ?a ?r) (agent_has_item ?a ?i))
    :effect (and (item_at ?i ?r) (not (agent_loaded ?a)) (not (agent_has_item ?a ?i)))
  )
)