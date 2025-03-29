from settings import settings

if settings:
    settings_dict = settings.dict()
    for location_name, location_settings in settings_dict.items():
        print(f"Location: {location_name}")
        print(f"  Start ID: {location_settings['start_ID']}")
        print(f"  End ID: {location_settings['end_ID']}")
        print(f"  Climb: {location_settings['climb']}")
else:
    print("Settings could not be loaded due to validation errors.")
