def is_valid_card_combination(cards):
    """
    Validate that the card combination is valid.
    :param cards: List of card objects
    :return: Boolean indicating if the combination is valid
    """
    seen_cards = set()
    for card in cards:
        if card in seen_cards:
            return False
        seen_cards.add(card)
    return True
