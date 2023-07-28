import pydealer
import time
import csv
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import altair as alt
import statistics

def card_to_integer(card):
    #Used to convert cards from pydealer package to a numeric value in order to calculate for BlackJack games
    #input: pydealer card
    #output: numeric value of given card
    if card.value in ["King", "Queen", "Jack"]:
        return 10
    elif card.value == "Ace":
        return 1
    else:
        return int(card.value)

def check_for_ace(hand):
    for card in hand:
        if card.value == "Ace":
            return True
    return False

def hand_sum(hand):
    #Used to determine the total value of a given hand
    #input: hand, list of card objects given to either dealer or player
    #output: int total value of hand
    if check_for_ace(hand):
        num_of_full_aces = 1
        sum_with_ace = [0,0]
        for card in hand:
            card_val = card_to_integer(card)
            if (card_val == 1):
                if num_of_full_aces > 0:
                    sum_with_ace[0] += 11
                    num_of_full_aces -= 1
                else:
                    sum_with_ace[0] += 1
                sum_with_ace[1] += 1
            else:
                sum_with_ace[0] += card_val
                sum_with_ace[1] += card_val
        return sum_with_ace
    else:
        sum_without_ace = 0
        for card in hand:
            card_val = card_to_integer(card)
            sum_without_ace += card_val
        return [sum_without_ace]

def pick_sum(totals):
    #Determine which of the sums of hands to return when there is an Ace in the hand
    if len(totals) == 1:
        return totals[0]
    else:
        sum_max = max(totals)
        sum_min = min(totals)
        if sum_max > 21 and sum_min > 21:
            return sum_min
        if sum_max > 21:
            return sum_min
        else:
            return sum_max

def running_hand_sum(hand):
    #Update the sum if more cards are dealt
    return pick_sum(hand_sum(hand))

def dealer_decision(hand):
    #Determine what the dealer will do based on their current hand
    action = "blank"
    total = running_hand_sum(hand)
    if total == 21 and len(hand) == 2:
        action = "blackjack"
    elif total >= 18:
        action = "stand"
    elif check_for_ace(hand) == True:
        if total <= 17:
            action = "hit"
        else:
            action = "stand"
    elif check_for_ace(hand) == False:
        if total >= 17:
            action = "stand"
        elif total <= 16:
            action = "hit"
    return action

def player_decision_1(hand, dealer_upcard):
    action = "blank"
    total = running_hand_sum(hand)
    dealer_upcard_value = card_to_integer(dealer_upcard)
    if total == 21 and len(hand) == 2:
        action = "blackjack"
    elif total in range(12, 17) and dealer_upcard_value in range (2,7):
        action = "stand"
    elif total == 16 and dealer_upcard_value == 10 and len(hand) > 2:
        action = "stand"
    elif total in range(1, 13):
        action = "hit"
    elif total in range(12, 17) and dealer_upcard_value in range(7,11):
        action = "hit"
    elif total in range(12, 17) and dealer_upcard_value == 1:
        action = "hit"
    elif check_for_ace(hand) == True:
        if total in range(18,22):
            action = "stand"
        else:
            action = "hit"
    elif check_for_ace(hand) == False:
        if total in range(17,22):
            action = "stand"
        else:
            action = "hit"
    elif total >= 20:
        if total == 21 and len(hand) == 2:
            action = "blackjack"
        else:
            action = "stand"
    return action

def result_index(result = 0, player_sum = [0], dealer_sum = 0, record = 0):
    dict_of_result = {"result": result, "player_sum": player_sum, "dealer_sum": dealer_sum, "record": record}
    return dict_of_result

def game_play(player_hand = None, dealer_hand = None, deck = None):
    if deck is None:
        deck = pydealer.Deck()
    if player_hand is None:
        player_hand = pydealer.Stack(cards = [deck.random_card(remove=True), deck.random_card(remove=True)])
    if dealer_hand is None:
        dealer_hand = pydealer.Stack(cards = [deck.random_card(remove=True), deck.random_card(remove=True)])
    player_sum = running_hand_sum(player_hand)
    dealer_sum = running_hand_sum(dealer_hand)
    player_move = player_decision_1(player_hand, dealer_hand[0])
    if player_move == "blackjack":
        if dealer_decision(dealer_hand) == "blackjack":
            return result_index(result = 0, player_sum = [21], dealer_sum = 21, record = "tie")
        return result_index(result = 1.5, player_sum = [21], dealer_sum = dealer_sum, record = "win")
    elif player_move == "stand":
        return finish_game(player_hand, dealer_hand, deck)
    elif player_move == "hit":
        player_hand.add(deck.random_card(remove=True))
        player_sum = running_hand_sum(player_hand)
        if player_sum > 21:
            return result_index(result = -1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "loss")
        player_move2 = player_decision_1(player_hand, dealer_hand[0])
        if player_move2 == "stand":
            return finish_game(player_hand, dealer_hand, deck)
        elif player_move2 == "hit":
            player_hand.add(deck.random_card(remove=True))
            player_sum = running_hand_sum(player_hand)
            if player_sum > 21:
                return result_index(result = -1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "loss")
            player_move3 = player_decision_1(player_hand, dealer_hand[0])
            if player_move3 == "stand":
                return finish_game(player_hand, dealer_hand, deck)
            elif player_move3 == "hit":
                player_hand.add(deck.random_card(remove=True))
                player_sum = running_hand_sum(player_hand)
                if player_sum > 21:
                    return result_index(result = -1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "loss")
                player_move4 = player_decision_1(player_hand, dealer_hand[0])
                if player_move4 == "stand":
                    return finish_game(player_hand, dealer_hand, deck)
                elif player_move4 == "hit":
                    player_hand.add(deck.random_card(remove=True))
                    player_sum = running_hand_sum(player_hand)
                    if player_sum > 21:
                        return result_index(result = -1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "loss")
                    player_move5 = player_decision_1(player_hand, dealer_hand[0])
                    if player_move5 == "stand":
                        return finish_game(player_hand, dealer_hand, deck)
                    elif player_move5 == "hit":
                        player_hand.add(deck.random_card(remove=True))
                        player_sum = running_hand_sum(player_hand)
                        if player_sum > 21:
                            return result_index(result = -1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "loss")
                        player_move6 = player_decision_1(player_hand, dealer_hand[0])
                        if player_move6 == "stand":
                            return finish_game(player_hand, dealer_hand, deck)

def finish_game(player_hand, dealer_hand, deck):
    player_sum = running_hand_sum(player_hand)
    dealer_move = dealer_decision(dealer_hand)
    player_move = player_decision_1(player_hand, dealer_hand[0])
    if dealer_move == "blackjack":
        player_sum = running_hand_sum(player_hand)
        dealer_sum = running_hand_sum(dealer_hand)
        return result_index(result = -1, player_sum = [player_sum], dealer_sum = 21, record = "loss")
    while player_move == "stand":
        if dealer_move == "hit":
            dealer_hand.add(deck.random_card(remove=True))
            dealer_sum = running_hand_sum(dealer_hand)
            if dealer_sum > 21:
                    return result_index(result = 1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "win")
            dealer_move2 = dealer_decision(dealer_hand)
            if dealer_move2 == "stand":
                player_sum = running_hand_sum(player_hand)
                dealer_sum = running_hand_sum(dealer_hand)
                bet_result = 0
                if dealer_sum > 21:
                    bet_result = 1
                    game_record = "win"
                elif player_sum > dealer_sum:
                    bet_result = 1
                    game_record = "win"
                elif dealer_sum > player_sum:
                    bet_result = -1
                    game_record = "loss"
                elif player_sum == dealer_sum:
                    bet_result = 0
                    game_record = "tie"
                return result_index(result = bet_result, player_sum = [player_sum], dealer_sum = dealer_sum, record = game_record)
            elif dealer_move2 == "hit":
                dealer_hand.add(deck.random_card(remove=True))
                dealer_sum = running_hand_sum(dealer_hand)
                if dealer_sum > 21:
                    return result_index(result = 1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "win")
                dealer_move3 = dealer_decision(dealer_hand)
                if dealer_move3 == "stand":
                    player_sum = running_hand_sum(player_hand)
                    dealer_sum = running_hand_sum(dealer_hand)
                    bet_result = 0
                    if dealer_sum > 21:
                        bet_result = 1
                        game_record = "win"
                    elif player_sum > dealer_sum:
                        bet_result = 1
                        game_record = "win"
                    elif dealer_sum > player_sum:
                        bet_result = -1
                        game_record = "loss"
                    elif player_sum == dealer_sum:
                        bet_result = 0
                        game_record = "tie"
                    return result_index(result = bet_result, player_sum = [player_sum], dealer_sum = dealer_sum, record = game_record)
                elif dealer_move3 == "hit":
                    dealer_hand.add(deck.random_card(remove=True))
                    dealer_sum = running_hand_sum(dealer_hand)
                    if dealer_sum > 21:
                        return result_index(result = 1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "win")
                    dealer_move4 = dealer_decision(dealer_hand)
                    if dealer_move4 == "stand":
                        player_sum = running_hand_sum(player_hand)
                        dealer_sum = running_hand_sum(dealer_hand)
                        bet_result = 0
                        if dealer_sum > 21:
                            bet_result = 1
                            game_record = "win"
                        elif player_sum > dealer_sum:
                            bet_result = 1
                            game_record = "win"
                        elif dealer_sum > player_sum:
                            bet_result = -1
                            game_record = "loss"
                        elif player_sum == dealer_sum:
                            bet_result = 0
                            game_record = "tie"
                        return result_index(result = bet_result, player_sum = [player_sum], dealer_sum = dealer_sum, record = game_record)
                    elif dealer_move4 == "hit":
                        dealer_hand.add(deck.random_card(remove=True))
                        dealer_sum = running_hand_sum(dealer_hand)
                        if dealer_sum > 21:
                            return result_index(result = 1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "win")
                        dealer_move5 = dealer_decision(dealer_hand)
                        if dealer_move5 == "stand":
                            player_sum = running_hand_sum(player_hand)
                            dealer_sum = running_hand_sum(dealer_hand)
                            bet_result = 0
                            if dealer_sum > 21:
                                bet_result = 1
                                game_record = "win"
                            elif player_sum > dealer_sum:
                                bet_result = 1
                                game_record = "win"
                            elif dealer_sum > player_sum:
                                bet_result = -1
                                game_record = "loss"
                            elif player_sum == dealer_sum:
                                bet_result = 0
                                game_record = "tie"
                            return result_index(result = bet_result, player_sum = [player_sum], dealer_sum = dealer_sum, record = game_record)
                        elif dealer_move5 == "hit":
                            dealer_hand.add(deck.random_card(remove=True))
                            dealer_sum = running_hand_sum(dealer_hand)
                            if dealer_sum > 21:
                                return result_index(result = 1, player_sum = [player_sum], dealer_sum = dealer_sum, record = "win")
                            dealer_move6 = dealer_decision(dealer_hand)
                            if dealer_move6 == "stand":
                                player_sum = running_hand_sum(player_hand)
                                dealer_sum = running_hand_sum(dealer_hand)
                                bet_result = 0
                                if dealer_sum > 21:
                                    bet_result = 1
                                    game_record = "win"
                                elif player_sum > dealer_sum:
                                    bet_result = 1
                                    game_record = "win"
                                elif dealer_sum > player_sum:
                                    bet_result = -1
                                    game_record = "loss"
                                elif player_sum == dealer_sum:
                                    bet_result = 0
                                    game_record = "tie"
                                return result_index(result = bet_result, player_sum = [player_sum], dealer_sum = dealer_sum, record = game_record)
        elif dealer_move == "stand":
            player_sum = running_hand_sum(player_hand)
            dealer_sum = running_hand_sum(dealer_hand)
            bet_result = 0
            if dealer_sum > 21:
                bet_result = 1
                game_record = "win"
            elif player_sum > dealer_sum:
                bet_result = 1
                game_record = "win"
            elif dealer_sum > player_sum:
                bet_result = -1
                game_record = "loss"
            elif player_sum == dealer_sum:
                bet_result = 0
                game_record = "tie"
            return result_index(result = bet_result, player_sum = [player_sum], dealer_sum = dealer_sum, record = game_record)


#Runs 100 rounds of 100 games and stores them as a list of dictionaries, the first range() will be the number of rounds and the second range() number will be the number of games per round
games = []
for i in range(100):
    results = []
    for x in range(100):
        result_dict = game_play()
        results.append(result_dict)
    games.append(results)

#Converts the list of dictionaries into a list of pandas dataframes
games_df = list()
for i in games:
    game = pd.DataFrame(i)
    games_df.append(game)

#Takes just the resulting pay out of each game and converts each next row into the cumulative sum of the previous game from the same round
data_for_plot = games_df[0].copy()
for idx, i in enumerate(games_df):
    data_for_plot[str(idx)] = games_df[idx]["result"].cumsum()

data_for_plot = data_for_plot.drop(["dealer_sum", "player_sum", "record", "result"], axis=1)

#Creates Visualizations
plt.hist(data_for_plot.tail(1).values.flatten().tolist(), bins=20)
plt.title("Histogram of Cumulative Sums of 100 Games (AddStrat3)")
plt.xlabel("Cumulative Sum of 100 Games")
plt.savefig('addstrat3hist')
columns = [str(x) for x in range(100)]
plot = data_for_plot.plot(y=columns, use_index=True, legend=None)
plt.plot
plt.title("Cumulative Sums of 100 Rounds of 100 Games (AddStrat3)")
plt.xlabel("Game Number")
plt.ylabel("Cumulative Sum")
plt.savefig('addstrat3')
print(statistics.pstdev(data_for_plot.tail(1).values.flatten().tolist()))
print(statistics.mean(data_for_plot.tail(1).values.flatten().tolist()))