d <- fromJSON('./refModule/json/costs-fixedPose96.json.raw') %>%
  gather(id, val) %>% 
  mutate(val = (val - min(val))/(max(val) - min(val)))
write(toJSON(d %>% spread(id, val)), './refModule/json/costs-fixedPose96.json')
