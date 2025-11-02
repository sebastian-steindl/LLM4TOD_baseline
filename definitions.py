from dataclasses import dataclass
from string import Template
from prompts import FewShotPrompt, SimpleTemplatePrompt

domain_prompt_check_response = """I want you to make sure your previous response was correct. Make sure to respond with exactly one word from this list: [restaurant, hotel, attraction, taxi, train] that describes what the customer is looking for. Make sure that you respond with the last domain that the conversation is about.
If your previous response was correct, just give me the same response again.

Domain:
"""
response_prompt_check_response = """I want you to make sure your previous response was correct. Your response should only include the fitting utterance as a response to the customer. Make sure to use the brackets exactly as defined earlier. Do not write code. Do not explain your decision. Do not start with 'Response:' or other introductions.
Never state that you can't do bookings directly, this is an imagined conversation where every booking attempt is succesful. Base your answer on the information given under state and database. 
Keep in mind that it should be a realistic dialogue flow where the customer is being helped as good as possible, but your response should follow the style given in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

If your previous response was correct, just give me the same response again.

response:
"""

MWZ_SLOTS_FOR_TEMPLATE = {
    "restaurant": ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"],
    "hotel": ["area", "internet", "parking", "stars", "type", "pricerange", "name", "booktime", "bookpeople", "bookstay"],
    "attraction": ["type", "area", "name"],
    "train": ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"],
    "taxi": ['departure', 'destination', 'leaveat', 'arriveby'],
    "hospital": ['department'],
    "bus": [],
}

MWZ_SLOT_PER_DOMAIN = {
    "restaurant": ["pricerange. Possible values: cheap, moderate, expensive",
                   "area. Possible values: centre, south, north, east, west",
                   "food",
                   "name",
                   "bookday",
                   "bookpeople",
                   "booktime"],
    "hotel": ["area. Possible values: east, centre, north, south, west",
              "internet. Possible values: yes, no",
              "parking. Possible values: yes, no",
              "stars. Possible values: 0, 1, 2, 3, 4, 5",
              "type.  Possible values: hotel, bed and breakfast, guest house",
              "pricerange. Possible values: cheap, moderate, expensive",
              "name",
              "booktime",
              "bookpeople",
              "bookstay"],
    "attraction": ["type. Possible values: museum, entertainment, college, multiple sports, nightclub, architecture, cinema, boat, theatre, park, concerthall, swimmingpool",
                   "area. Possible values: north, east, west, south, centre ",
                   "name"],
    "train": ["arriveby",
              "leaveat",
              "day. Possible values: monday, tuesday, wednesday, thursday, friday, saturday, sunday",
              "departure",
              "destination",
              "bookpeople"],
    "taxi": ['departure',
             'destination',
             'leaveat',
             'arriveby'],
    "hospital": ['department'],
    "bus": [],
}

state_prompt_check_response = Template("""I want you to make sure your previous response was correct. Make sure that your response follows the format entity:'value'-entity:'value'. Your response should not use any additional words or remarks. If a slot has no value, just leave its value as '?'. Your response should not invent any slots or values, stick to the conversation as closely as possible.
Make sure everything is logical.
If your previous response was correct, just give me the same response again.

Update the following state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:

state: $state
""")


multiwoz_domain_prompt = SimpleTemplatePrompt(template="""
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
 - restaurant
 - hotel
 - attraction
 - taxi
 - train
Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.

-------------------
Example1:
Customer: I need a cheap place to eat
Assistant: We have several not expensive places available. What food are you interested in?
Customer: Chinese food.

Domain: restaurant

-------

Example 2:
Customer: I also need a hotel in the north.
Assistant: Ok, can I offer you the Molly's place?
Customer: What is the address?

Domain: hotel

---------

Example 3:
Customer: What is the address?
Assistant: It's 123 Northfolk Road.
Customer: That's all. I also need a train from London.

Domain: train
""

Now complete the following example, answering with exactly one word:
{}
{}
Domain:""", args_order=["history", "utterance"])

"""
######################
FEW SHOT
######################
"""

@dataclass
class FewShotRestaurantDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from the last two utterances of the converstation according to examples.
Focus only on the values mentioned in the last two utterances.
Capture pair "entity:'value'" separated by colon.
Separate entity:'value' pairs by hyphens.

Values that should be captured are the following. For some I give you the only possible values:
 - "pricerange" that specifies the price range of the restaurant. Only possible values for pricerange: cheap, moderate, expensive
 - "area" that specifies the area where the restaurant is located. Only possible values for area: centre, south, north, east, west
 - "food" that specifies the type of food the restaurant serves
 - "name" that specifies the name of the restaurant
 - "bookday" that specifies the day of the booking
 - "booktime" that specifies the time of the booking. Time will be in format hh:mm
 - "bookpeople" that specifies for how many people is the booking made
 
Do not capture any other entities or values! Do not invent any slots or values, stick to the conversation as closely as possible.
If no value is specified, leave the value at '?'.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state: {}""", # pricerange:'?'-area:'?'-food:'?'-name:'?'-bookday:'?'-booktime:'?'-bookpeople:'?'
                                    args_order=["history", "utterance", "state"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a restaurant. The conversation should first find a suitable restaurant and then offer to make a reservation.
The customer can search for a restaurant by the parameters area, food, or pricerange. If the customer does not give you this information, you can ask him for it.

There is also a number of restaurants in the database currently corresponding to the user's request.
If the database returns 0 restaurants, you can offer to search with different parameters.
If the database returns at least one restaurants, include [choice] in your response as a placeholder for the number of restaurants.
If the database returns a restaurant and you are asked for information, respond with the placeholder [choice] for the amount and these placeholders for pieces of information: [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode]. 

To make a reservation, the following information needs to be represented in the state or what the customer said, otherwise you need to ask the customer to give you this information: bookday (= for which day),  booktime (= at which time), and bookpeople (= how many guests).
If you have enough information to book or make a reservation, you can ask the customer if you should book for him. If the customer tells you to book or make a reservation, include [ref] in your answer, and act as if the booking was successful.
Provide the placeholder in brackets as I have shown you. Do not use the actual information, just the placeholders.

For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

---------
{}{}
---------
Now write a response for the following conversation:
context: {}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"]


@dataclass
class FewShotHotelDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from the last two utterances of the converstation according to examples.
Focus only on the values mentioned in the last two utterances.
Capture pair "entity:'value'" separated by colon.
Separate entity:'value' pairs by hyphens.

Values that should be captured are the following. For some I give you the only possible values:
 - "area" that specifies the area where the hotel is located. Only possible values for area: east, centre, north, south, west
 - "internet" that specifies if the hotel has internet. Only possible values for internet: yes, no
 - "parking" that specifies if the hotel has parking. Only possible values for parking: yes, no
 - "stars" that specifies the number of stars the hotel has. Only possible values for stars: 0, 1, 2, 3, 4, 5
 - "type" that specifies the type of the hotel. Only possible values for type: hotel, bed and breakfast, guest house
 - "pricerange" that specifies the price range of the hotel. Only possible values for pricerange: cheap, moderate, expensive
 - "name" that specifies name of the hotel
 - "bookstay" specifies length of the stay
 - "bookday" specifies the day of the booking
 - "bookpeople" specifies how many people should be booked for.
 
Do not capture any other entities or values! Do not invent any slots or values, stick to the conversation as closely as possible.
If not specified, leave the value empty.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state: {}""", # state: area:'?'-internet:'?'-parking:'?'-stars:'?'-type:'?'-pricerange:'?'-name:'?'-bookstay:'?'-bookday:'?'-bookpeople:'?'""",
                                    args_order=["history", "utterance", "state"])
    response_prompt = FewShotPrompt(template="""
    Definition: You are an assistant that helps people to book a hotel. The conversation should first find a suitable hotel and then offer to make a reservation.
The customer can search for a hotel by the parameters name, area, parking, internet availability, or price.
There is also a number of hotels in the database currently corresponding to the user's request.
If the database returns 0 hotels, you can offer to search with different parameters.
If the database returns at least one hotel, include [choice] in your response as a placeholder for the number of hotels and [type] for the type.
If the database returns a hotel, and you are asked for information, respond with the placeholder [choice] for the amount and these placeholders for pieces of information: [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode].
To make a reservation, the following information needs to be represented in the state or what the customer said, otherwise you need to ask the customer to give you this information: bookstay (= for how many days),  booktime (= at which time), and bookpeople (= how many guests).

If you have enough information to book or make a reservation, you can ask the customer if you should book for him. If the customer tells you to book or make a reservation, include [ref] in your answer, and act as if the booking was successful.
Provide the placeholder in brackets as I have shown you. Do not use the actual information, just the placeholders.


For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

---------
{}{}
---------
Now write a response for the following conversation:
context: {}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange", "name", "booktime", "bookpeople", "bookstay"]


@dataclass
class FewShotTrainDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from the last two utterances of the converstation according to examples.
Focus only on the values mentioned in the last two utterances.
Capture pair "entity:'value'" separated by colon.
Separate entity:'value' pairs by hyphens.

Values that should be captured are the following. For some I give you the only possible values:
 - "arriveby" that specifies what time the train should arrive. Time will be in format hh:mm
 - "leaveat" that specifies what time the train should leave. Time will be in format hh:mm
 - "day" that specifies what day the train should leave. Only possible values for day: monday, tuesday, wednesday, thursday, friday, saturday, sunday
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "bookpeople" that specifies how many people the booking is for
 
Do not capture any other entities or values! Do not invent any slots or values, stick to the conversation as closely as possible.
If no value is specified, leave the value at '?'.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state: {}""", #state: arriveby:'?'-leaveat:'?'-day:'?'-departure:'?'-destination:'?'-bookpeople:'?'""",
                                    args_order=["history", "utterance", "state"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a train connection.
The customer can search for a train by the parameters arriveby, leaveat, day, departure or destination. If the customer does not give you this information, you can ask him for it.
There is also a number of trains in the database currently corresponding to the user's request.
If the database returns 0 trains, you can offer to search with different parameters.
If the database returns at least one train, include [choice] in your response as a placeholder for the number of trains.
If you are talking about one specific train, include [trainid] in your response. If you are asked for information, respond with the placeholder [trainid] and these placeholders for pieces of information: [arriveby], [leaveat], [destination] or [departure].

To make a reservation, the following information needs to be represented in the state or what the customer said, otherwise you need to ask the customer to give you this information: day (= for which day),  departure (= from where), destination (= to where), and bookpeople (= how many guests).
If you have enough information to book or make a reservation, you can ask the customer if you should book for him. If the customer tells you to book or make a reservation, include [ref] and [price] in your answer, and act as if the booking was successful.
Provide the placeholder in brackets as I have shown you. Do not use the actual information, just the placeholders.

For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

---------
{}{}
---------
Now write a response for the following conversation:
context: {}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"]


@dataclass
class FewShotTaxiDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from the last two utterances of the converstation according to examples.
Focus only on the values mentioned in the last two utterances.
Capture pair "entity:'value'" separated by colon.
Separate entity:'value' pairs by hyphens.

Values that should be captured are:
 - "arriveby" that specifies what time the taxi should arrive. Time will be in format hh:mm
 - "leaveat" that specifies what time the taxi should leave. Time will be in format hh:mm
 - "departure" that specifies the departure of the taxi drive
 - "destination" that specifies the destination of the taxi drive
 
Do not capture any other entities or values! Do not invent any slots or values, stick to the conversation as closely as possible.
If no value is specified, leave the value at '?'.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state: {}""",# state: arriveby:'?'-leaveat:'?'-departure:'?'-destination:'?'""",
                                    args_order=["history", "utterance", "state"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a taxi.
To make a booking, the following information needs to be represented in the state or what the customer said, otherwise you need to ask the customer to give you this information:  either leaveat(= time to leave) or arriveby(= time to arrive),  departure (= from where), destination (= to where).
If you have this information, you can book the taxi. Then include the placeholders [type] and [phone] in your response.
Provide the placeholder in brackets as I have shown you. Do not use the actual information, just the placeholders.

For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

---------
{}{}
---------
Now write a response for the following conversation:
{}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveat', 'arriveby']



@dataclass
class FewShotHospitalDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from the last two utterances of the converstation according to examples.
Focus only on the values mentioned in the last two utterances.
Capture pair "entity:'value'" separated by colon.
Separate entity:'value' pairs by hyphens.

Values that should be captured are the following.
 - "department" that specifies the department of interest
 
Do not capture any other entities or values! Do not invent any slots or values, stick to the conversation as closely as possible.
If no value is specified, leave the value at '?'.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state: {}""",#state: department:'?'""",
                                    args_order=["history", "utterance", "state"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a hospital.


For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

---------
{}{}
---------
Now write a response for the following conversation:
{}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['department']


@dataclass
class FewShotAttractionDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from the last two utterances of the converstation according to examples.
Focus only on the values mentioned in the last two utterances.
Capture pair "entity:'value'" separated by colon.
Separate entity:'value' pairs by hyphens.

Values that should be captured are the following. For some I give you the only possible values:
 - "type" that specifies the type of attraction. Only possible values for type: museum, entertainment, college, multiple sports, nightclub, architecture, cinema, boat, theatre, park, concerthall, swimmingpool
 - "area" that specifies the area where the attraction is located. Only possible values for area: north, east, west, south, centre
 - "name" that specifies the name of the attraction

Do not capture any other entities or values! Do not invent any slots or values, stick to the conversation as closely as possible.
If no value is specified, leave the value at '?'.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state: {}""",#state: type:'?'-area:'?'-name:'?'""",
                                    args_order=["history", "utterance", "state"])
    response_prompt = FewShotPrompt(template="""
    Definition: You are an assistant that helps people to find an attraction. The conversation should find a suitable attraction, and offer information about it.
The customer can search for an attraction by the parameters name, area, or type.

If the database returns 0 attractions, you can offer to search with different parameters.
If the database returns at least one attractions, include [choice] in your response as a placeholder for the number of attractions as well as [entrancefee].
If you are talking about a specific attraction, and you are asked for information, respond with the placeholders [type], [area], [name], [entrancefee] [attraction_address], [attraction_phone], [attraction_postcode].
Provide the placeholder in brackets as I have shown you. Do not use the actual information, just the placeholders.

For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.
---------
{}{}
---------
Now write a response for the following conversation:
{}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area", "name"]



@dataclass
class FewShotBusDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

If not specified, leave the value empty.
------
{}{}
---------
For the following example, the state is not updated. Respond with the updated state so that it correctly represents the information given in the last two utterances, one from the customer and one from the assistant:
context: {}
Customer: {}

state:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a bus.
For your response, mimick the style of the assistant in the examples.
If a topic is complete, because a booking is finished, canceled or for other reasons, ask if the customer needs anything else. If the customer says in his last utterance that he does not need anything else and / or says good bye, you only thank him and respond with a polite farewell.

---------
{}{}
---------
Now write a response for the following conversation:
context: {}
Customer: {}

Use this information as demonstrated in the examples above:
state: {}
database: {}

response:""",                args_order=["history", "utterance", "state", "database"])
    expected_slots = []


"""
######################
ZERO SHOT
######################
"""


@dataclass
class ZeroShotRestaurantDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture each pair as "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
 - "name" that is the name of the restaurant
Do not capture any other values!
If not specified, leave the value empty.
{}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [address].
If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked.
If booking, provide [ref] in the answer.

{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food", "name"]


@dataclass
class ZeroShotHotelDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
Do not capture any other values!
If not specified, leave the value empty.

{}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
    Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotel in the database currently corresponding to the user's request.
If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [address].
If booking, provide [ref] in the answer.

{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange"]


@dataclass
class ZeroShotTrainDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
Do not capture any other values!
If not specified, leave the value empty

{}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, provide [arriveby], [leaveat] or [departure] if asked.
Do not provide real entities in the response! Just provide entity name in brackets, like [duration] or [price].
If booking, provide [ref] in the answer.

{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveBy", "leaveAt", "day", "departure", "destination"]


@dataclass
class ZeroShotTaxiDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)

{}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to book a taxi.
Do not provide real entities in the response! Just provide entity name in brackets, like [color] or [type].

{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveAt', 'arriveBy', 'date']



@dataclass
class ZeroShotHospitalDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.

{}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a hospital.

{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []



@dataclass
class ZeroShotBusDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.

{}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a bus.

{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []


@dataclass
class ZeroShotAttractionDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
Do not capture any other values!
If not specified, leave the value empty.

{}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
    Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
Do not provide real entities in the response! Just provide entity name in brackets, like [address] or [name].
If you find a hotel, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked.

{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area"]


MW_FEW_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": FewShotRestaurantDefinition,
    "hotel": FewShotHotelDefinition,
    "attraction": FewShotAttractionDefinition,
    "train": FewShotTrainDefinition,
    "taxi": FewShotTaxiDefinition,
    "hospital": FewShotHospitalDefinition,
    "bus": FewShotBusDefinition,
}


MW_ZERO_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": ZeroShotRestaurantDefinition,
    "hotel": ZeroShotHotelDefinition,
    "attraction": ZeroShotAttractionDefinition,
    "train": ZeroShotTrainDefinition,
    "taxi": ZeroShotTaxiDefinition,
    "hospital": ZeroShotHospitalDefinition,
    "bus": ZeroShotBusDefinition,
}