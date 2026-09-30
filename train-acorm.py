import numpy as np
import torch
import os
import pickle
#from agent.wqmixcltd3 import MADDPG
from agent.acormtd3 import MADDPG
#from agent.qmixtd3 import MADDPG
#from args.argument_u4m36 import get_args
#from args.argument_u4m30 import get_args
from args.argument_u4m20 import get_args
#from args.argument_u4m25 import get_args
#from args.argument_u3m30 import get_args
#from args.argument import get_args
from env.mec_scenery import Scenario
from env.mec_env import MECEnv
from utils import visualizer
from utils.logger0 import Logger
from utils.visualizer import Visualizer


def train():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    print(f"使用设备: {device}")

    scenario = Scenario()
    world = scenario.make_world(args)
    env = MECEnv(args, world, scenario.reset_world, scenario.get_obs, scenario.get_reward)
    
    maddpg = MADDPG(args, device)

    # =====================================================================
    # 【核心修改 1：加载阈值法训练好的下层模型】
    # =====================================================================
    # 请将这里的路径替换为你保存的阈值下层模型的真实路径
    pretrained_lower_path = '/kaggle/input/models/wangkaila/acorm45/pytorch/default/1/40/best_model.pkl'
    #pretrained_lower_path = 'rule/best_model_u4m20.pkl'
    maddpg.load_lower_model(pretrained_lower_path)
    
    # 可选：如果你彻底不想让下层再变动，可以关掉下层的动作噪声
    #maddpg.noise.sigma = 0.2  # 给极小的噪声即可，或者直接设为 0
    #maddpg.noise.target_sigma = 0.1  # 给极小的噪声即可，或者直接设为 0

    logger = Logger(log_dir='logs')
    visualizer = Visualizer(save_dir='results')

    print("\n开始训练...\n")

    total_steps = 0
    best_upper = 0
    best_completion = 0
    best_upper_penalty = -100

    agent_embed_pretrain_epoch, recl_pretrain_epoch = 0, 0
    pretrain_agent_embed_loss, pretrain_recl_loss = [], []

    for episode in range(args.num_episodes):
        env.reset()
        maddpg.upper_agent.init_hidden_states()

        episode_rewards = []
        episode_upper_rewards = []
        episode_upper_penalty = []
        current_u_traj = []
        current_t_traj = []
        completion_rate = []
        avg_delay = []
        avg_energy = []
        avg_fly_energy = []
        avg_uncovered = []

        #upper_critic_losses = []
        #upper_actor_losses = []
        recl_losses = []
        qmix_losses = []
        role_decoder_losses = []
        critic1_loss = []
        critic2_loss = []
        service_loss = []

        roles = []
        best_uav_battery = []
        best_term = []

        last_a = np.full((args.num_uavs, ), -1)  # 动作：角色0/1、水平速度、方向、卸载比

        dw = False
        for large_step in range(args.episode_length):
            #upper_reward = 0.0
            upper_state, upper_obs = env.start_large_step()
            #改为ACORM选择角色，根据当前观测obs和上一步动作last_a_n
            bats = np.array([u.get_battery_ratio() for u in world.uavs])
            #print(upper_obs.shape)  # 2,5
            #print(last_a.shape)   # 2,
            role, role_embeddings = maddpg.select_roles(upper_obs, last_a)
            #print(role.shape)     # 同last_a的形状

            upper_penalty = 0.0
            for uav in world.uavs:
                uav.role = role[uav.uav_id]
                if uav.role == 0:   # and bat_u > 0.9: 高电量选择充电
                    #upper_reward -= 1.5    # 2.0-1.5
                    bat_u = uav.get_battery_ratio()
                    overcharge_penalty = max(0, (bat_u - 0.80) * 5.0) 
                    upper_penalty -= overcharge_penalty
            # roles, role_embeddings = self.agent.choose_role(obs_n, last_a_n, epsilon=epsilon)
            # 开始大时间步追踪
            #print(env.world.current_large_step)
            maddpg.start_large_step(env.world.current_large_step - 1, upper_state, upper_obs, last_a, role)
            last_a = role.copy()
            large_step_rewards = []
            roles.append(role.flatten())
            best_uav_battery.append(bats)
            best_term.append(env.world.current_large_step_tasks)

            for small_step in range(args.steps_per_large):
                obs, state = env.start_small_step()
                # 1. 记录 UAV 位置
                u_pos = np.array([[u.location.x, u.location.y, u.location.z] for u in world.uavs])
                current_u_traj.append(u_pos)

                # 2. 记录 终端 位置 (假设有 num_terminals 个)
                # 注意：终端数量固定吗？如果是动态生成的，这里需要用 padding 或者 list
                # 假设数量固定为 args.num_terminals
                t_pos = np.array([[t.location.x, t.location.y, t.has_task] for t in world.terminals])
                #print(t_pos)  # 1./0. 表示有无任务生成
                current_t_traj.append(t_pos)

                #根据智能体嵌入和角色嵌入以及角色来选择任务卸载动作，维度变化
                actions, raw_actions = maddpg.select_actions(obs, role, explore=True)
                next_obs, next_state, reward, done, info = env.step(actions)

                store = False
                for u in world.uavs:
                    if u.role == 1:  # 服务角色
                        store = True
                        break

                #store = all(u.role == 1 for u in world.uavs)
                if store:
                    maddpg.store_lower_transition(obs, state, raw_actions, reward * 0.1, next_obs, next_state, float(done))

                #maddpg.store_lower_transition(obs, state, raw_actions, reward * 0.1, next_obs, next_state, float(done))

                # 记录小步奖励reward 13
                maddpg.store_small_step_reward(reward * 0.1)
                large_step_rewards.append(reward * 0.1)

                # 这部分改变了，原先设置是一整套流程下来，上下层一起更新，但现在要为下层单独创建一个策略网络，用独立价值网络更新
                # 更新下层网络
                if episode >= 800:
                    if total_steps % 20 == 0:   # 减缓下层更新频率
                        if maddpg.lower_buffer.size > args.min_batch_size:
                            #if total_steps % 5 == 0:   # 减缓下层更新频率
                            losses = maddpg.update_lower()
                            if losses[0] is not None:
                                logger.log_losses(*losses)
                                critic1_loss.append(float(losses[0]))
                                critic2_loss.append(float(losses[1]))
                            if losses[2] is not None:
                                service_loss.append(float(losses[2]))

                total_steps += 1
                obs = next_obs
                state = next_state

                if done:
                    break

            # ====== 大时间步结束 ======
            # 获取新的上层状态,滑动窗口没有更新，但若在这更新了，就又会重复更新
            #next_upper_state = env.get_upper_state()
            #next_upper_obs = env.get_upper_obs()
            # 结束大时间步，存储上层经验
            for uav in world.uavs:
                bat = uav.get_battery_ratio()

                # 1. 极严重惩罚：坠机 (必须保留，但不要大到让它绝望)
                if uav.is_crashed:
                    upper_penalty -= 2.5  # 抵消掉一个大步的收益 ， 10-7-5-4-2.5-3.0-2.5-3.5-2.5

                elif 0.1 < bat < 0.2:
                    upper_penalty += (0.2 - bat) * 5.0

                # 2. 柔性电量引导 (不再用 -5 这么大的数)
                '''elif bat < 0.05:   # 0.2-0.05
                    # 当电量从 20% 掉到 5% 时，惩罚从 0 线性增加到 -2.0
                    # 这样即使两架飞机都低电量，也就 -4.0，不会盖过 +10 的任务奖励
                    penalty = (0.05 - bat) * 10.0  # (0.2-0.05)*13.3 = 2.0-----13.3-10.0
                    upper_penalty -= penalty'''

            if (large_step + 1) == args.episode_length:
                dw = True

            # ACORM算法中有时序信息的处理，需要上一步隐藏状态，所以它是整个序列的即整个大时间步经验作为一个条存储
            # 也就是在上面小时间步存储时同时进行大时间步的存储，并带上当前的小时间步数
            # 结束大时间步，存储上层经验
            upper_reward = maddpg.end_large_step(upper_penalty, dw)
            # self.replay_buffer.store_transition(step=buffer_step, obs_n=obs_n, s=state, last_a_n=last_a_n, a_n=actions, role_n=roles, r=reward, dw=done)


            # 更新上层网络
            #if maddpg.upper_buffer.size > args.min_batch_size:
            #    maddpg.upper_agent.train()
            maddpg.upper_agent.epsilon = maddpg.upper_agent.epsilon - args.epsilon_decay if maddpg.upper_agent.epsilon - args.epsilon_decay > args.epsilon_min else args.epsilon_min

            episode_rewards.append(np.mean(large_step_rewards))
            episode_upper_rewards.append(upper_reward * 0.1)
            episode_upper_penalty.append(upper_penalty)
            completion_rate.append(np.mean(env.world.task_completion_rate))
            #print(env.world.average_delay)
            avg_delay.append(np.mean(env.world.average_delay))
            avg_energy.append(np.mean(env.world.average_energy))
            avg_fly_energy.append(np.mean(env.world.uav_fly_energy))
            avg_uncovered.append(np.mean(env.world.uncovered_terminals_count))

        if env.world.current_large_step >= args.episode_length:
            env.world.task_history_window.append(env.world.current_large_step_tasks)
            #print("处理最后一步滑动窗口")
            next_upper_state = env.get_upper_state()
            next_upper_obs = env.get_upper_obs()
            maddpg.upper_buffer.store_last_step(env.world.current_large_step, next_upper_state, next_upper_obs, roles[-1])

        if agent_embed_pretrain_epoch < args.agent_embed_pretrain_epochs:
            if maddpg.upper_buffer.current_size >= args.upper_batch_size:
                agent_embed_pretrain_epoch += 1
                #print(f"预训练嵌入")
                agent_embedding_loss = maddpg.upper_agent.pretrain_agent_embedding(maddpg.upper_buffer)
                pretrain_agent_embed_loss.append(agent_embedding_loss)
        else:
            if recl_pretrain_epoch < args.recl_pretrain_epochs:
                recl_pretrain_epoch += 1
                #print(f"预训练对比学习")
                recl_loss = maddpg.upper_agent.pretrain_recl(maddpg.upper_buffer)
                pretrain_recl_loss.append(recl_loss)
            else:
                if maddpg.upper_buffer.current_size >= args.upper_batch_size:
                    recl_loss, qmix_loss, role_decoder_loss = maddpg.upper_agent.train(maddpg.upper_buffer)  # Training
                    #maddpg.upper_agent.epsilon = max(args.epsilon_min, maddpg.upper_agent.epsilon * args.epsilon_decay)
                    if qmix_loss is not None:
                        logger.log_recl_losses(qmix_loss, recl_loss, role_decoder_loss)
                        recl_losses.append(recl_loss)
                        qmix_losses.append(qmix_loss)
                        role_decoder_losses.append(role_decoder_loss)


        # 一个 Episode 结束后衰减一次（更新频率每一个大时间步）
        maddpg.noise.sigma = max(maddpg.noise.min_sigma, maddpg.noise.sigma * maddpg.noise.delay)
        #maddpg.noise.sigma = max(0.05, maddpg.noise.sigma * 0.999)

        avg_episode_reward = np.mean(episode_rewards)
        completion_rate = np.mean(completion_rate)
        avg_upper_reward = np.mean(episode_upper_rewards)
        avg_upper_penalty = np.mean(episode_upper_penalty)
        avg_delay = np.mean(avg_delay)
        avg_energy = np.mean(avg_energy)
        avg_fly_energy = np.mean(avg_fly_energy)
        avg_uncovered = np.mean(avg_uncovered)
        #avg_upper_critic_loss = np.mean(upper_critic_losses) if upper_critic_losses else 0
        #avg_upper_actor_loss = np.mean(upper_actor_losses) if upper_actor_losses else 0
        avg_recl_loss = np.mean(recl_losses) if recl_losses else 0
        avg_qmix_loss = np.mean(qmix_losses) if qmix_losses else 0
        avg_role_decoder_loss = np.mean(role_decoder_losses) if role_decoder_losses else 0
        avg_critic1_loss = np.mean(critic1_loss) if critic1_loss else 0
        avg_critic2_loss = np.mean(critic2_loss) if critic2_loss else 0
        avg_service_loss = np.mean(service_loss) if service_loss else 0

        # 新增：实时追加保存本轮完整调度数据
        logger.log_episode_history(episode, roles, best_uav_battery, best_term)

        # 记录到logger
        logger.log_episode(
            episode,
            avg_upper_reward,
            avg_episode_reward,
            completion_rate,
            avg_delay,
            avg_energy,
            avg_fly_energy,
            avg_uncovered,
            #env.uav_locations,  # 传入完整轨迹
            #avg_upper_actor_loss,
            #avg_upper_critic_loss,
            avg_recl_loss,
            avg_qmix_loss,
            avg_role_decoder_loss,
            avg_critic1_loss,
            avg_critic2_loss,
            avg_service_loss
        )

        if avg_upper_penalty > best_upper_penalty:
            best_upper_penalty = avg_upper_penalty
            best_roles = np.stack([r.flatten() for r in roles])
            best_battery = np.array(best_uav_battery).copy()
            best_term = np.array(best_term).copy()
            logger.best_roles = best_roles
            logger.best_battery = best_battery
            logger.best_term = best_term
            logger.save_role()
            os.makedirs('checkpoints/0705cl', exist_ok=True)
            maddpg.save('checkpoints/0705cl/best_model.pkl')
            print(f"\n 新的最佳模型! 完成率: {completion_rate:.2%}")
        elif avg_upper_penalty == best_upper_penalty:
            if avg_upper_reward > best_upper:
                best_upper = avg_upper_reward
                best_roles = np.stack([r.flatten() for r in roles])
                best_battery = np.array(best_uav_battery).copy()
                best_term = np.array(best_term).copy()
                logger.best_roles = best_roles
                logger.best_battery = best_battery
                logger.best_term = best_term
                logger.save_role()
                os.makedirs('checkpoints/0705cl', exist_ok=True)
                maddpg.save('checkpoints/0705cl/best_model.pkl')
                print(f"\n 新的最佳模型! 完成率: {completion_rate:.2%}")

        # 保存最佳模型
        '''if avg_upper_reward > best_upper:
            best_upper = avg_upper_reward
            # 深拷贝轨迹数据
            # best_trajectory = np.array(current_episode_trajectory).copy()
            #best_roles = np.array(roles).copy()
            best_roles = np.stack([r.flatten() for r in roles])
            best_battery = np.array(best_uav_battery).copy()
            best_term = np.array(best_term).copy()
            logger.best_roles = best_roles
            logger.best_battery = best_battery
            logger.best_term = best_term
            logger.save_role()
            os.makedirs('checkpoints', exist_ok=True)
            maddpg.save('checkpoints/best_model.pkl')
            print(f" 新的最佳模型! 完成率: {completion_rate:.2%}")'''

        if avg_episode_reward > best_completion:
            best_completion = avg_episode_reward
            best_uav_traj = np.array(current_u_traj).copy()
            best_term_traj = np.array(current_t_traj).copy()
            logger.best_trajectory = best_uav_traj
            logger.best_terminals = best_term_traj
            logger.save_traj()
            # 只绘制最佳轨迹
            if len(best_uav_traj) > 0:
                visualizer.plot_uav_trajectories(
                    uav_trajectories=best_uav_traj,
                    episode_idx="best_model",  # 标记这是最佳模型的轨迹
                    args=args,  # 传入 args 以获取地图尺寸和充电站位置
                    terminal_trajectories=best_term_traj,  # 传入终端轨迹
                    save_name='best_trajectory_visualization.png'
                )
            #print("\n生成最佳轨迹可视化...")


        # 定期保存模型
        '''if (episode + 1) % args.save_interval == 0:
            os.makedirs('checkpoints/0625cl', exist_ok=True)
            maddpg.save(f'checkpoints/0625cl/model_episode_{episode + 1}.pkl')'''

        # 定期保存模型
        if (episode + 1) % args.save_interval == 0:
            os.makedirs('checkpoints', exist_ok=True)
            maddpg.save_checkpoint(filepath=f'checkpoints/latest_checkpoint_M{args.num_terminals}.pkl',
                                   buffer_filepath=f'checkpoints/latest_buffer_M{args.num_terminals}.pkl')
            print(f"[{episode}] 已保存最新 Checkpoint，用于断点续训。")

        # 打印统计信息
        if (episode + 1) % args.print_interval == 0:
            stats = logger.get_statistics()
            print(f"\nEpisode {episode}/{args.num_episodes}")
            #print(f"  平均奖励: {stats['avg_reward']:.2f}")
            print(f"  任务完成率: {stats['avg_completion_rate']:.2%}")
            print(f"  平均延迟: {stats['avg_delay']:.2f}s")
            print(f"  覆盖终端: {args.num_terminals - stats['avg_uncovered']:.2f}/{args.num_terminals}")
            #print(f"  下层平均奖励: {avg_episode_reward:.2f}")
            #print(f"  上层累积奖励: {avg_upper_reward:.2f}")
            print(f"  上层Epsilon: {maddpg.upper_agent.epsilon:.4f}")

            # 打印UAV状态
            for uav in world.uavs:
                role_str = '充电' if uav.role == 0 else '服务'
                print(f"    UAV{uav.uav_id}: ({uav.location.x:.0f},{uav.location.y:.0f},{uav.location.z:.0f}), "
                      f"电量{uav.get_battery_ratio():.0%}, {role_str}")
            print(f"{'=' * 80}")

    print("\n训练完成!")
    print("=" * 80)

    # 保存最终模型
    #maddpg.save('checkpoints/0625cl/final_model.pkl')

    # 保存数据
    data_dir = logger.save_data()

    # 生成可视化
    #print("\n生成可视化结果...")
    #visualizer.plot_training_curves(data_dir)

    print("\n所有结果已保存!")
    print(f"  模型: checkpoints/")
    print(f"  数据: {data_dir}")
    print(f"  图像: results/")


if __name__ == '__main__':
    train()