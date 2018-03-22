#- coding: utf-8 -*-
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser


class Info(models.Model):

    status = models.SmallIntegerField(default=0) # -1: found, 0: published, Z: number of msg 

    kind = models.IntegerField()  # -1: unknown, 1: woman, 2: kids, 3: elder, 4: teenages, 5: else

    title = models.CharField(max_length=100)

    name = models.CharField(max_length=30)
    sex  = models.IntegerField()            # 1: male, 0: female
    age  = models.IntegerField(null=True)

    date = models.DateTimeField()

    location_1 = models.CharField(max_length=100, default='')
    location_2 = models.CharField(max_length=100, default='')
    location_3 = models.CharField(max_length=100, default='')
    location_4 = models.CharField(max_length=100, default='')
    location_all = models.CharField(max_length=300, default='')

    # features outline
    height  = models.IntegerField(null=True)
    weight  = models.IntegerField(null=True)


    cloth_color = models.CharField(max_length=20, null=True)
    hair_color  = models.CharField(max_length=20, null=True)
    hair_length = models.CharField(max_length=5,  null=True)
    glass_flag  = models.BooleanField(default=False)
    beard_flag  = models.BooleanField(default=False)

    disability_flag = models.BooleanField(default=False)
    disability_head = models.BooleanField(default=False)
    disability_leg  = models.BooleanField(default=False)
    disability_face = models.BooleanField(default=False)
    alzheimer_flag  = models.BooleanField(default=False)
    insane_flag     = models.BooleanField(default=False)

    more_info   = models.CharField(max_length=400, null=True)

    release_date = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True

class Photo(models.Model):
    url = models.CharField(max_length=200)
    location = models.CharField(max_length=200)

    size = models.DecimalField(max_digits=4, decimal_places=1, null=True)
    width  = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    height = models.DecimalField(max_digits=6, decimal_places=2, null=True)

    release_date = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class LostPhoto(Photo):

    def __unicode__(self):
        return self.id
    class Meta:
        db_table = u'lost_photo'
        
class FindPhoto(Photo):

    class Meta:
        db_table = u'find_photo'

class LostInfo(Info):

    user = models.ForeignKey(User, unique=False)

    contract_name = models.CharField(max_length=30)
    telephone     = models.CharField(max_length=20)
    reward        = models.DecimalField(max_digits=9, decimal_places=2, default=0)

    photos = models.ManyToManyField(LostPhoto)


    def __unicode__(self):
        return self.name
    class Meta:
        db_table = u'lost_info'

class FindInfo(Info):

    #user = models.ForeignKey(User, unique=False)

    contract_name = models.CharField(max_length=30)
    telephone = models.CharField(max_length=20)

    photos = models.ManyToManyField(FindPhoto)

    class Meta:
        db_table = u'find_info'

class Msg(models.Model):

    find_info = models.ForeignKey(FindInfo, unique=False)
    lost_info = models.ForeignKey(LostInfo, unique=False)


    read_flag = models.BooleanField(default=False)

    class Meta:
        db_table = u'msg'

class Notify(models.Model):

    user = models.ForeignKey(User, unique=False)

    msgs = models.ManyToManyField(Msg)
    
    def __get_unread_num(self):
        unread_num = 0
        read_num = 0
        for msg in self.msgs.all():
            if msg.read_flag == False:
                unread_num += 1
            else:
                read_num += 1
        return [unread_num,read_num]
        
    num = property(__get_unread_num)

    class Meta:
        db_table = u'notify'




