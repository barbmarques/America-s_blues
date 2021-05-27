
| column_name  | description                                                           | key                          | dtype  |
|-------------|-----------------------------------------------------------------------|------------------------------|--------|
| age | Age of victim in years  |  | object |
| gender | Gender of victim |    | object |
| race|  Race of victim |   | object |
| date  | Date of incident |                        | datetime  |
| city | City where incident occurred|      | object  |
| state |  State where incident occurred    |    | float |
| zipcode | Zip code where incident occurred  |  | object |
| county| County where incident occurred                           |                              | object |
| agency_responsible  |  Identifies the agency represented at the incident (police department, sheriff's office, marshall's office, etc.)    |     | object  |
| cause_of_death | Victim's cause of death: (gunshot, taser, vehicle, physical restraint, beaten, baton, bomb, police dog, asphyxiation, pepper spray, chemical agent, other)         |                              | object  |
| description_of_circumstances |Brief description of the incident's offense and outcome  | | object |
| official_disposition  |Indicates the status of charges, if they were filed against the officer.| | object |
| criminal_charges_filed|Indicates whether or not charges were filed against the officer for the killing.   | | object |
| mental_illness   |Indicates whether the victim was identified as having a mental illness |                              | object  |
| armed_unarmed_status |A person was coded as 'unarmed' if there were not holding any objects or weapons when killed, if they were holding personal items that were not used to attack others (ex: cellphone, cane, etc.), if they were holding a toy weapon (ex. BB gun, pellet gun, air rifle, toy sword), if they were an innocent bystander or hostage, or a person or motorist killed after being intentionally hit by a police car or as a result of hitting police stop sticks during a pursuit | | object |
| alleged_weapon |Alleged weapon of the victim. |         | object |
| alleged_threat_lvl | Indicates whether or not the officer was allegedly attacked by the victim.  |   | object |
| fleeing      | Indicates whether or not the victim was fleeing at the time of the incident.  |   | object |
| body_camera|Indicates whether or not police body camera footage of the event is available. |                              | object |
| geography  | The location of the incident is classified as either: suburban, urban, rural, undetermined. |                              | object |
| encounter_type_draft | Classifies each incidence as: violent crime, person with a weapon, domestic disturbance, traffic stop, mental health/welfare check, other non-violent offense, other crimes again people, or unknown.     || object |
| initial_reported_reason_for_encounter_draft| Reason for initial encounter with police.                   |                              | object |
|race_of_officers_involved_draft  | Indicates race of the officer involved.   |                              | object |
| known_past_shootings_of_Officer_draft | Indicates whether the officer involved has been involved in past shootings.             |                              | object  |
| call_for_service_draft |Indicates whether or not police were responding to a call for service regarding the incidence. ||object|
