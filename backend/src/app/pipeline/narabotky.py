# from app.sources.demo_source import DemoPredictionSource
# DataSourceFactory.register("demo", DemoPredictionSource)
# DataSourceFactory.register("demo", DemoPredictionSource)

# sources = []

# for conf in config.sources:
#     name = conf["type"]
#     params = {k: v for k, v in conf.items() if k != "type"}
#     sources.append(DataSourceFactory.create(name, **params))

# predictions = []
# for source in sources:
#     predictions.extend(source.get_predictions())