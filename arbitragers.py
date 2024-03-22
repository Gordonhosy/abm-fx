import mesa
from resources import *
import numpy as np
import random
import math
from banks import *

class agent_arbitrager(mesa.Agent):
    '''
    Arbitragers:
    - Arbitragers finds arbitrage opportunities based on their tradable markets
    - Their trading volume is limited by their small asset size
    - If bid-ask crossed, they sell at highest bid and buy at lowest ask
    '''
    
    def __init__(self, agent_id, model, pos, moore, currencyA, currencyB, cost_currencyA, cost_currencyB, vision):
        super().__init__(agent_id, model)
        self.pos = pos
        self.moore = moore
        self.currencyA = currencyA
        self.currencyB = currencyB
        self.cost_currencyA = cost_currencyA
        self.cost_currencyB = cost_currencyB
        self.vision = vision
        self.traded_prices = []
        self.traded_banks = []
        self.traded_amount = []
    
    
    def is_occupied_banks(self, pos):
        '''
        Function to check if a cell is occupied by a bank
        '''
        if pos == self.pos:
            return False
        
        # check if this cell includes something that is mutually exclusive
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            
            ##### Need to amend the agent corporate later ##### 
            if isinstance(obj, agent_bank):
                return True
        return False    
            
        
    def get_banks(self, pos):
        '''
        Return the bank agent of the postition if any
        '''
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, agent_bank):
                return obj
        return None
        
        
    def pay_costs(self):
        '''
        Function for corporate to pay money each step
        So far it is not incorporated
        '''
        self.currencyA -= self.cost_currencyA
        self.currencyB -= self.cost_currencyB
     
    
    def if_bankrupt(self):
        '''
        Function to check if someone is bankrupted, and remove the agent
        '''
        if (self.currencyA < 0) | (self.currencyB < 0):
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
    
    

    def arbitrage_lob(self):
        '''
        Function to arbitrage based on the arbitrager's tradable LOBs
        '''
                
        neighbor_banks = [self.get_banks(pos) for pos in self.model.grid.get_neighborhood(self.pos, self.moore, True, self.vision) if self.is_occupied_banks(pos)]
        
        if len(neighbor_banks) == 0:
            return
        
        ##### capacity need to be changed #####
        capacity = self.currencyA
        
        bid_book_bank, ask_book_bank = self.observe_lob(neighbor_banks)
        self.do_arb(capacity, bid_book_bank, ask_book_bank)
        
        return
   


    def observe_lob(self, neighbor_banks):
        '''
        Return the tradable LOBs for the arbitrager
        '''
        bid_book_bank = {}
        for bank in neighbor_banks:
            for bid in bank.bid_book:
                if bid[0] in bid_book_bank:
                    bid_book_bank[bid[0]].append([bid[1], bank])
                else:
                    bid_book_bank[bid[0]] = [[bid[1], bank]]

        ask_book_bank = {}
        for bank in neighbor_banks:
            for ask in bank.ask_book:
                if ask[0] in ask_book_bank:
                    ask_book_bank[ask[0]].append([ask[1], bank])
                else:
                    ask_book_bank[ask[0]] = [[ask[1], bank]]
        
        return bid_book_bank, ask_book_bank
        
        
    def do_arb(self, capacity, bid_book_bank, ask_book_bank):
        '''
        Execute the arbitrage
        '''
        ask_from_low = sorted(list(ask_book_bank.keys()))
        bid_from_high = sorted(list(bid_book_bank.keys()), reverse = True)
        
        # when all surroundings banks die
        if (len(ask_from_low) == 0) | (len(bid_from_high) == 0):
            return
        
        next_ask = next(iter(ask_from_low))
        next_bid = next(iter(bid_from_high))
        
        while ((next_ask < next_bid) & (capacity > 0)):
            ask_amount = sum([pair[0] for pair in ask_book_bank[next_ask]])
            top_bid_amount = sum([pair[0] for pair in bid_book_bank[next_bid]])
            
            # cleared the top of ask
            if ask_amount < top_bid_amount: 
                self.execute_arb_full(next_ask, ask_book_bank, next_bid, bid_book_bank, ask_amount)
                ask_from_low.pop(0)
                capacity -= ask_amount
                
            # cleared the top of bid
            elif ask_amount > top_bid_amount:
                self.execute_arb_part(next_ask, ask_book_bank, next_bid, bid_book_bank, top_bid_amount)
                bid_from_high.pop(0)
                capacity -= top_bid_amount
                
            # cleared both bars
            else:
                self.execute_arb_full(next_ask, ask_book_bank, next_bid, bid_book_bank, ask_amount)
                ask_from_low.pop(0)
                bid_from_high.pop(0)
                capacity -= ask_amount
                
            next_ask = next(iter(ask_from_low))
            next_bid = next(iter(bid_from_high))
                
    def execute_arb_full(self, ask_price, ask_book_bank, bid_price, bid_book_bank, amount):
        '''
        Execute the arbitrage, clear all ask top and part of bid top
        '''
        
        # buy the whole bar of lower ask
        for volu_bank in ask_book_bank[ask_price]:
            
            self.currencyA += volu_bank[0]
            self.currencyB -= volu_bank[0]*ask_price
            volu_bank[1].currencyA -= volu_bank[0]
            volu_bank[1].currencyB += volu_bank[0]*ask_price

            self.traded_prices.append(ask_price)
            self.traded_banks.append(volu_bank[1].unique_id)
            self.traded_amount.append(volu_bank[0])

            volu_bank[1].arbed_amount.append(-volu_bank[0])
            volu_bank[1].arbed_prices.append(ask_price)
            volu_bank[1].arbed_desks.append(self.unique_id)
            
            volu_bank[0] = 0
        
        # sell at higher bid
        # randomly choose banks to sell at
        random_volu_bank = [volu_bank for volu_bank in bid_book_bank[bid_price]]
        random.shuffle(random_volu_bank)
        
        for volu_bank in random_volu_bank:
            
            if volu_bank[0] == 0:
                pass
            elif volu_bank[0] < amount:
                self.currencyA -= volu_bank[0]
                self.currencyB += volu_bank[0]*bid_price
                volu_bank[1].currencyA += volu_bank[0]
                volu_bank[1].currencyB -= volu_bank[0]*bid_price

                self.traded_prices.append(bid_price)
                self.traded_banks.append(volu_bank[1].unique_id)
                self.traded_amount.append(-volu_bank[0])

                volu_bank[1].arbed_amount.append(volu_bank[0])
                volu_bank[1].arbed_prices.append(bid_price)
                volu_bank[1].arbed_desks.append(self.unique_id)
                
                amount -= volu_bank[0]
                volu_bank[0] = 0
                
            else:
                self.currencyA -= amount
                self.currencyB += amount*bid_price
                volu_bank[1].currencyA += amount
                volu_bank[1].currencyB -= amount*bid_price

                self.traded_prices.append(bid_price)
                self.traded_banks.append(volu_bank[1].unique_id)
                self.traded_amount.append(-amount)

                volu_bank[1].arbed_amount.append(amount)
                volu_bank[1].arbed_prices.append(bid_price)
                volu_bank[1].arbed_desks.append(self.unique_id)
                
                volu_bank[0] -= amount
                
                break
        
        
    def execute_arb_part(self, ask_price, ask_book_bank, bid_price, bid_book_bank, amount):
        '''
        Execute the arbitrage, clear part of ask top and all of bid top
        '''
        
        # randomly buy lower ask
        random_volu_bank = [volu_bank for volu_bank in ask_book_bank[ask_price]]
        random.shuffle(random_volu_bank)
        
        for volu_bank in random_volu_bank:
            
            if volu_bank[0] == 0:
                pass
            
            elif volu_bank[0] < amount:
                self.currencyA += volu_bank[0]
                self.currencyB -= volu_bank[0]*ask_price
                volu_bank[1].currencyA -= volu_bank[0]
                volu_bank[1].currencyB += volu_bank[0]*ask_price

                self.traded_prices.append(ask_price)
                self.traded_banks.append(volu_bank[1].unique_id)
                self.traded_amount.append(volu_bank[0])

                volu_bank[1].arbed_amount.append(-volu_bank[0])
                volu_bank[1].arbed_prices.append(ask_price)
                volu_bank[1].arbed_desks.append(self.unique_id)
                
                amount -= volu_bank[0]
                volu_bank[0] = 0
                
            else:
                self.currencyA += amount
                self.currencyB -= amount*ask_price
                volu_bank[1].currencyA -= amount
                volu_bank[1].currencyB += amount*ask_price

                self.traded_prices.append(ask_price)
                self.traded_banks.append(volu_bank[1].unique_id)
                self.traded_amount.append(amount)

                volu_bank[1].arbed_amount.append(-amount)
                volu_bank[1].arbed_prices.append(ask_price)
                volu_bank[1].arbed_desks.append(self.unique_id)
                
                volu_bank[0] -= amount
                
                break
        
        # sell the whole top of bid
        for volu_bank in bid_book_bank[bid_price]:
            
            self.currencyA -= volu_bank[0]
            self.currencyB += volu_bank[0]*bid_price
            volu_bank[1].currencyA += volu_bank[0]
            volu_bank[1].currencyB -= volu_bank[0]*bid_price

            self.traded_prices.append(bid_price)
            self.traded_banks.append(volu_bank[1].unique_id)
            self.traded_amount.append(-volu_bank[0])

            volu_bank[1].arbed_amount.append(+volu_bank[0])
            volu_bank[1].arbed_prices.append(bid_price)
            volu_bank[1].arbed_desks.append(self.unique_id)
            
            volu_bank[0] = 0