from pydantic import BaseModel


class ParkRangerScenario(BaseModel):
    request: str


train_scenarios = [
    ParkRangerScenario(request="I live in New York, NY. Where can I find bears?"),
    ParkRangerScenario(
        request="Hey! I'm in San Francisco, CA. Where can I find turtles near me?"
    ),
    ParkRangerScenario(
        request="I'm visiting Denver, Colorado. Where can I spot elk in the wild?"
    ),
    ParkRangerScenario(
        request="Looking for eagles near Seattle, Washington. Any recommendations?"
    ),
    ParkRangerScenario(
        request="I'm in Miami, Florida. Where's the best place to see alligators?"
    ),
    ParkRangerScenario(request="Currently in Austin, Texas. Where can I find bats?"),
    ParkRangerScenario(
        request="I'm staying in Portland, Oregon. Where might I see deer?"
    ),
    ParkRangerScenario(
        request="Visiting Boston, Massachusetts. Any good spots for whale watching?"
    ),
    ParkRangerScenario(
        request="I'm in Phoenix, Arizona. Where can I observe desert wildlife like coyotes?"
    ),
    ParkRangerScenario(
        request="In Chicago, Illinois. Where's the best place to see migrating birds?"
    ),
    ParkRangerScenario(
        request="I'm in Nashville, Tennessee. Where can I find wild turkeys?"
    ),
    ParkRangerScenario(
        request="Currently in Salt Lake City, Utah. Where might I spot mountain goats?"
    ),
    ParkRangerScenario(
        request="I'm visiting Atlanta, Georgia. Any places to see wild deer?"
    ),
    ParkRangerScenario(request="In Las Vegas, Nevada. Where can I find bighorn sheep?"),
    ParkRangerScenario(
        request="I'm in Minneapolis, Minnesota. Where's good for spotting loons?"
    ),
    ParkRangerScenario(
        request="Visiting New Orleans, Louisiana. Where can I see pelicans?"
    ),
    ParkRangerScenario(
        request="I'm in Kansas City, Missouri. Where might I find wild prairie dogs?"
    ),
    ParkRangerScenario(
        request="Currently in Anchorage, Alaska. Where can I observe moose safely?"
    ),
    ParkRangerScenario(
        request="I'm in Honolulu, Hawaii. Where's the best place to see monk seals?"
    ),
    ParkRangerScenario(
        request="Visiting Burlington, Vermont. Where can I spot black bears?"
    ),
    ParkRangerScenario(
        request="I'm in Boise, Idaho. Where might I see pronghorn antelope?"
    ),
    ParkRangerScenario(
        request="Currently in Richmond, Virginia. Where can I find wild foxes?"
    ),
    ParkRangerScenario(
        request="I'm in San Diego, California. Where's good for spotting sea lions?"
    ),
]


val_scenarios = [
    ParkRangerScenario(
        request="I'm in Sacramento, California. Where can I find wild otters?"
    ),
    ParkRangerScenario(
        request="Currently in Tampa, Florida. Where's the best place to see manatees?"
    ),
    ParkRangerScenario(
        request="I'm visiting Jackson, Wyoming. Where might I spot wolves?"
    ),
    ParkRangerScenario(
        request="In Pittsburgh, Pennsylvania. Where can I observe wild raccoons?"
    ),
    ParkRangerScenario(
        request="I'm in Spokane, Washington. Where can I find wild salmon runs?"
    ),
    ParkRangerScenario(
        request="Currently in Raleigh, North Carolina. Where might I see wild opossums?"
    ),
    ParkRangerScenario(
        request="I'm visiting Billings, Montana. Where can I spot bison?"
    ),
    ParkRangerScenario(
        request="In Columbus, Ohio. Where's good for watching migrating butterflies?"
    ),
]
