# Feature Catalog

| Feature | Entity | Type | Description | Definition |
|---|---|---|---|---|
| `user_activity_count` | user | int | Number of interactions by the user | `count(interactions) as of cutoff` |
| `user_avg_rating` | user | float | Mean rating the user gives (global mean if cold-start) | `mean(rating)` |
| `user_rating_std` | user | float | Spread of the user's ratings | `std(rating)` |
| `user_positive_rate` | user | float | Share of the user's interactions rated >= threshold | `mean(rating >= 4)` |
| `preferred_category` | user | str | Category with most positive interactions (ties -> most interactions) | `argmax_cat(n_positive, n_all)` |
| `preferred_category_share` | user | float | Share of user's interactions that fall in preferred category | `n_all[pref] / n_all` |
| `days_since_last_interaction` | user | float | Days between cutoff and user's latest interaction (tenure if none) | `as_of - max(timestamp)` |
| `user_tenure_days` | user | float | Days since signup | `as_of - signup_date` |
| `age` | user | int | User age (imputed with median if missing) | `clean_users.age` |
| `item_popularity` | item | int | Number of interactions with the item | `count(interactions)` |
| `item_positive_count` | item | int | Number of positive interactions | `count(rating >= 4)` |
| `item_avg_rating` | item | float | Bayesian-smoothed mean rating | `(n*mean + m*global_mean)/(n+m)` |
| `item_rating_std` | item | float | Spread of ratings received | `std(rating)` |
| `item_popularity_percentile` | item | float | Percentile rank of item_popularity in [0,1] | `rank(item_popularity)/N` |
| `item_age_years` | item | int | Years between release and cutoff year | `as_of.year - release_year` |
| `item_days_since_last_interaction` | item | float | Recency of the item's latest interaction | `as_of - max(timestamp)` |
| `category` | item | str | Item category | `clean_items.category` |
| `price` | item | float | Item price | `clean_items.price` |
| `user_item_category_match` | interaction | int | 1 if item category == user's preferred category | `category == preferred_category` |
| `user_category_affinity` | interaction | float | Share of user's history in the item's category | `n_user_cat / n_user` |
| `baseline_rating_estimate` | interaction | float | Bias-based rating estimate used by the rating model | `user_avg + item_avg - global_mean (clipped 1..5)` |
